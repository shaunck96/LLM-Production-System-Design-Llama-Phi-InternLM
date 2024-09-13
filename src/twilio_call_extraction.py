import requests
import logging
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import pandas as pd
from datetime import datetime, timedelta
import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_date, lit, udf, first, min, last
from pyspark.sql.types import StringType
from concurrent.futures import as_completed
from typing import Dict, Any, Set
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import concurrent.futures
from tqdm import tqdm

spark = SparkSession.builder.appName("Optimize Production Code").getOrCreate()


class SafeQueryDSL:
    def __init__(self):
        self.query_parts = []
        self.where_conditions = []
        self.params = []

    def select(self, *columns):
        self.query_parts.append(("SELECT", columns))
        return self

    def from_table(self, database, table):
        self.query_parts.append(("FROM", f"{database}.{table}"))
        return self

    def join(self, table, condition):
        self.query_parts.append(("JOIN", (table, condition)))
        return self

    def where(self, condition):
        self.where_conditions.append(condition)
        return self

    def add_param(self, param):
        self.params.append(param)
        return self

    def build(self):
        query = ""
        for part_type, part_value in self.query_parts:
            if part_type == "SELECT":
                query += f"SELECT {', '.join(part_value)} "
            elif part_type == "FROM":
                query += f"FROM {part_value} "
            elif part_type == "JOIN":
                query += f"JOIN {part_value[0]} ON {part_value[1]} "
        
        if self.where_conditions:
            query += f"WHERE {' AND '.join(self.where_conditions)} "
        
        return query.strip(), self.params


class TwilioCallExtractor:
    def __init__(self, config_path: str):
        """
        Initializes a TwilioCallExtractor instance.

        Args:
            config_path (str): The path to the configuration file.
        """
        self.config = self.load_config(config_path)
        self.bill_creds = self.load_config("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/bill_account_secrets.json")
        self.setup_dates()
        self.setup_paths()
        self.setup_client()
        self.logger = self.setup_logger()
        self.setup_whitelists()
        self.dsl = SafeQueryDSL()

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Loads the configuration from a JSON file.

        Args:
            config_path (str): The path to the JSON configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration dictionary.

        Raises:
            ValueError: If the configuration file is not found.
        """
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise ValueError(f"Configuration file not found: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {str(e)}")

    def setup_dates(self) -> None:
        """
        Initializes the start and end dates for the data extraction process.

        Sets `end_date` to the day before the current date, `start_date` to one year before `end_date`, 
        and `end_date_str` to the formatted string of `end_date`.
        """        
        self.end_date = datetime.strptime(self.bill_creds['end_date'], "%Y-%m-%d").date() - timedelta(days=1)
        json_start_date = datetime.strptime(self.bill_creds['start_date'], "%Y-%m-%d").date()
        self.start_date = (self.end_date - timedelta(days=365))
        self.start_date_str = self.start_date.strftime("%Y-%m-%d")
        self.end_date_str = self.end_date.strftime('%Y_%m_%d')

    def setup_paths(self) -> None:
        """
        Sets up the file paths for data and output directories based on configuration.

        Reads configuration values to initialize `data_path`, `output_path`, `bill_account_df_path`, 
        and `calls_metadata_path`. Uses default paths if configuration values are not provided.
        """
        self.data_path = self.config.get("audio_directory", "/default/path")
        self.output_path = self.config.get("metadata_output_path", "/default/output")
        self.bill_account_df_path = self.config.get("call_sid_list_path", "/default/call_sid_list.csv")
        self.calls_metadata_path = self.config.get("metadata_output_path", "/default/call_sid_list.csv")

    def setup_client(self) -> None:
        """
        Initializes the Twilio client using credentials from a JSON file.

        Reads Twilio credentials from a JSON file and initializes the `Client` instance. 
        Raises a `ValueError` if required configuration keys are missing or if client initialization fails.
        """
        try:
            with open("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/Production_Code/credentials/twilio_credentials.json", "r") as f:
                creds = json.load(f)

            self.account_sid = creds['account_sid'] 
            self.api_key = creds['api_key'] 
            self.api_secret = creds['api_secret']

            self.client = Client(self.api_key, self.api_secret, self.account_sid)

        except KeyError as e:
            raise ValueError(f"Missing Twilio configuration key: {str(e)}")
        except TwilioRestException as e:
            raise ValueError(f"Failed to initialize Twilio client: {str(e)}")

    def setup_logger(self) -> logging.Logger:
        """
        Configures the logger for the Twilio extractor.

        Sets up a logger with INFO level and configures it to write log messages to a file specified 
        in the configuration. The log file is named "twilio_extractor.log".
        
        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        log_path = self.config.get('log_path', '/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/logs/twilio/')
        #os.makedirs(log_path, exist_ok=True)
        handler = logging.FileHandler(f"twilio_extractor.log")
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def setup_whitelists(self) -> None:
        """
        Initializes sets of approved tables and columns for validation.

        Sets `approved_tables` to a set of valid table names and `approved_columns` to a set of valid column names.
        """
        self.approved_tables: Set[str] = {
            "csda.tasksegments", "csda.callHeader", "dfIntent", 
            "dfAudioResponseFiltered", "dfTransferReasonFiltered", "dfUnion", "Intent"
        }
        self.approved_columns: Set[str] = {
            "taskId", "agentConnectTimestamp", "taskURL", "accountId", "conversationId", 
            "timeStamp", "timestamp", "eventMessage_recognizerResult_text", "eventMessage_topIntent", 
            "eventMessage_topIntentScore", "channelId", "event_Name", "event_Type", 
            "brd_ActivityId", "brd_Text", "recognizerResult_text", "recognizerResult_intent_value",
            "recognizerResult_intent_score", "USER_RESPONSE", "INTENT", "INTENT_SCORE", "IVR_QUESTION",
            "User_Response", "Intent", "Intent_Score"
        }

    def validate_identifier(self, identifier: str, approved_set: Set[str]) -> str:
        """
        Validates an identifier against an approved set.

        Args:
            identifier (str): The identifier to validate.
            approved_set (Set[str]): The set of approved identifiers.

        Returns:
            str: The identifier enclosed in backticks if valid.

        Raises:
            ValueError: If the identifier is not in the approved set.
        """
        if identifier not in approved_set:
            raise ValueError(f"Unauthorized identifier: {identifier}")
        return f"`{identifier}`"

    def setup_df(self) -> None:
        """
        Loads and processes the DataFrame containing call SID data.

        Reads a parquet file into a DataFrame, renames a column, and sets `task_ids` to the first 200 task IDs.
        
        Raises:
            Exception: If an error occurs during file reading or processing.
        """
        try:
            call_sid_list = pd.read_parquet(self.bill_account_df_path)
            call_sid_list.rename(columns={'TASK_ID': 'taskId'}, inplace=True)
            self.task_ids = call_sid_list['taskId'].tolist()[:200]
        except Exception as e:
            self.logger.error(f"Error in setup_df: {str(e)}")
            raise

    @staticmethod
    def extract_recording_sid(url: str) -> str:
        """
        Extracts the recording SID from a URL.

        Args:
            url (str): The URL from which to extract the recording SID.

        Returns:
            str: The extracted recording SID, or an empty string if the URL is invalid.
        """
        return url.rsplit('/', 1)[-1] if url and isinstance(url, str) else ""
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10),
           retry=retry_if_exception_type(RequestException))
    def download_recording_single(self, rsid):
        """
        Downloads a single recording from Twilio.

        Args:
            rsid (str): The recording SID.

        Raises:
            RequestException: If the recording fetch or download fails.
        """
        try:
            recording_details = self.client.recordings(rsid).fetch()
            duration = recording_details.duration
            url = f'https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Recordings/{rsid}.mp3'
            file_path = f'/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/data/audio_files/{rsid}.mp3'
            response = requests.get(url, auth=(self.api_key, self.api_secret), stream=True, timeout=60)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Downloaded {file_path}")
            return file_path
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                self.logger.error(f"Authentication failed for recording {rsid}. Please check your Twilio credentials.")
            else:
                self.logger.error(f"HTTP error occurred while downloading recording {rsid}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to download recording {rsid}: {str(e)}")
        return "Not Downloaded"  

    def customer_metadata_extraction(self) -> pd.DataFrame:
        """
        Extracts customer metadata from a data source and returns it as a DataFrame.

        This method performs the following steps:
        1. Initializes Spark session.
        2. Retrieves credentials and configuration settings from secrets.
        3. Configures Spark with Azure Data Lake storage settings.
        4. (Additional processing steps would be added here.)

        Returns:
            pd.DataFrame: A DataFrame containing the extracted customer metadata.

        Raises:
            Exception: If any error occurs during metadata extraction or Spark setup.
        """        
        print("Starting Metadata Extraction")

        try:
            YEAR, MONTH = str(self.start_date_str).split('-')[0],self.start_date_str.split('-')[1]

            spark = SparkSession.builder.appName("Optimize Production Code").getOrCreate()

            adanexprDBricksID = dbutils.secrets.get(scope="pplz-key-adanexpr", key="Azure-SP-ADANEXPR-DBricks-ID")
            adanexprDBricksPWD = dbutils.secrets.get(scope="pplz-key-adanexpr", key="Azure-SP-ADANEXPR-DBricks-PWD")
            pplTenantID = dbutils.secrets.get(scope="pplz-key-adanexpr", key="tenant-id-adanexpr")

            dpcoreStorageAcct = dbutils.secrets.get(scope="pplz-key-adanexpr", key="storage-account-dpcore")
            dpcoreDBricksID = dbutils.secrets.get(scope="pplz-key-adanexpr", key="Azure-SP-ADANEXPR-DPCore-DBricks-ID")
            dpcoreDBricksPWD = dbutils.secrets.get(scope="pplz-key-adanexpr", key="Azure-SP-ADANEXPR-DPCore-DBricks-PWD")

            spark.conf.set("fs.azure.enable.check.access", "false")

            spark.conf.set(f"fs.azure.account.auth.type.{dpcoreStorageAcct}.dfs.core.windows.net", "OAuth")
            spark.conf.set(f"fs.azure.account.hns.enabled.{dpcoreStorageAcct}.dfs.core.windows.net", "true")
            spark.conf.set(f"fs.azure.account.oauth.provider.type.{dpcoreStorageAcct}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
            spark.conf.set(f"fs.azure.account.oauth2.client.id.{dpcoreStorageAcct}.dfs.core.windows.net", dpcoreDBricksID)
            spark.conf.set(f"fs.azure.account.oauth2.client.secret.{dpcoreStorageAcct}.dfs.core.windows.net", dpcoreDBricksPWD)
            spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{dpcoreStorageAcct}.dfs.core.windows.net", f"https://login.microsoftonline.com/{pplTenantID}/oauth2/token")

            AUDIO_RESPONSE_PATH = f"abfss://curated@{dpcoreStorageAcct}.dfs.core.windows.net/customer/omex/activitylog/current_view/audio_response/{YEAR}/{MONTH}/*/*/"
            TRANSFER_REASON_PATH = f"abfss://curated@{dpcoreStorageAcct}.dfs.core.windows.net/customer/omex/activitylog/current_view/transfertoagent_response/{YEAR}/{MONTH}/*/*/"
            INTENT_PATH = f"abfss://curated@{dpcoreStorageAcct}.dfs.core.windows.net/customer/omex/activitylog/current_view/nlp_result/{YEAR}/{MONTH}/*/*/"

            # Read data
            dfAudioResponse = spark.read.parquet(AUDIO_RESPONSE_PATH)
            dfTransferReason = spark.read.parquet(TRANSFER_REASON_PATH)
            dfIntent = spark.read.parquet(INTENT_PATH)

            date_filter = (to_date(col("timestamp")) >= lit(self.start_date)) & (to_date(col("timestamp")) <= lit(self.end_date))
            channel_filter = col("channelId") == "TwilioVoice"

            dfAudioResponseFiltered = dfAudioResponse.filter(channel_filter & date_filter)
            dfTransferReasonFiltered = dfTransferReason.filter(channel_filter & date_filter)
            dfIntentFiltered = dfIntent.filter(channel_filter & date_filter)

            approved_columns = [self.validate_identifier(col, self.approved_columns) for col in 
                                ["accountId", "conversationId", "timeStamp", "eventMessage_recognizerResult_text", 
                                 "eventMessage_topIntent", "eventMessage_topIntentScore"]]
            
            dfIntentPrepared = dfIntentFiltered.select(*approved_columns)

            dfUnion = (
                dfIntentPrepared.select(
                    col("accountId"), col("conversationId"), 
                    col("eventMessage_recognizerResult_text").alias("recognizerResult_text"),
                    col("eventMessage_topIntent").alias("recognizerResult_intent_value"),
                    col("eventMessage_topIntentScore").alias("recognizerResult_intent_score"),
                    col("timeStamp").alias("timestamp"),
                    lit(None).alias("event_Name"), lit(None).alias("event_Type"),
                    lit(None).alias("brd_ActivityId"), lit(None).alias("brd_Text")
                )
                .union(
                    dfTransferReasonFiltered.select(
                        lit(None).alias("accountId"), col("conversationId"),
                        lit(None).alias("recognizerResult_text"), lit(None).alias("recognizerResult_intent_value"),
                        lit(None).alias("recognizerResult_intent_score"), col("timestamp"),
                        col("event_Name"), col("event_Type"),
                        lit(None).alias("brd_ActivityId"), lit(None).alias("brd_Text")
                    )
                )
                .union(
                    dfAudioResponseFiltered.select(
                        lit(None).alias("accountId"), col("conversationId"),
                        lit(None).alias("recognizerResult_text"), lit(None).alias("recognizerResult_intent_value"),
                        lit(None).alias("recognizerResult_intent_score"), col("timestamp"),
                        lit(None).alias("event_Name"), lit(None).alias("event_Type"),
                        col("brd_ActivityId"), col("brd_Text")
                    )
                )
            ).orderBy("timestamp")

            Intent = (
                dfUnion.alias("df")
                .join(
                    dfUnion.filter(
                        (col("recognizerResult_intent_value") != "didnt_understand") &
                        (col("recognizerResult_text").isNotNull()) &
                        (col("recognizerResult_text") != " ")
                    )
                    .groupBy("conversationId")
                    .agg(
                        first("recognizerResult_text").alias("recognizerResult_text"),
                        first("recognizerResult_intent_value").alias("recognizerResult_intent_value"),
                        first("recognizerResult_intent_score").alias("recognizerResult_intent_score"),
                        min("timestamp").alias("min_timestamp")
                    )
                    .alias("min"),
                    (col("df.conversationId") == col("min.conversationId")) & (col("min.min_timestamp") > col("df.timestamp")),
                    "inner"
                )
                .select(
                    col("df.conversationId"),
                    col("df.timestamp").alias("timeStamp"),
                    col("min.recognizerResult_text").alias("USER_RESPONSE"),
                    col("min.recognizerResult_intent_value").alias("INTENT"),
                    col("min.recognizerResult_intent_score").alias("INTENT_SCORE"),
                    col("df.brd_Text").alias("IVR_QUESTION")
                )
                .orderBy("df.timestamp")
            )

            dfIntent2 = (
                Intent.groupBy("conversationId")
                .agg(
                    first("USER_RESPONSE").alias("User_Response"),
                    first("INTENT").alias("Intent"),
                    first("INTENT_SCORE").alias("Intent_Score"),
                    last("IVR_QUESTION").alias("IVR_Question")
                )
            )

            dfIntent2 = dfIntent2.toPandas()
            dfIntent2.rename(columns={"conversationId": "callSid"}, inplace=True)
            print("Metadata Extraction Complete")
            return dfIntent2
        except Exception as e:
            self.logger.error(f"Error in customer_metadata_extraction: {str(e)}")
            raise

    def recording_download_trigger(self) -> pd.DataFrame:
        """
        Trigger the download of recordings based on task IDs and process the data.

        This method performs the following steps:
        1. Validates the table identifier.
        2. Creates a temporary view of task IDs.
        3. Builds and executes a SQL query using SafeQueryDSL to retrieve task segments.
        4. Extracts recording SIDs from task URLs.
        5. Downloads recordings using multi-threading.
        6. Merges the download results with the original DataFrame.
        7. Extracts and merges customer metadata.
        8. Saves the resulting metadata to a CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the merged metadata for the processed recordings.

        Raises:
            Exception: If any error occurs during the process.
        """
        try:
            table_name = self.validate_identifier("csda.tasksegments", self.approved_tables)
            
            print(f"Validated table name: {table_name}")  # Debugging print

            # Create a temporary view of the task IDs
            task_ids_df = spark.createDataFrame([(id,) for id in self.task_ids], ["taskId"])
            task_ids_df.createOrReplaceTempView("task_ids_temp")

            # Register the extract_recording_sid function as a UDF
            extract_recording_sid_udf = udf(self.extract_recording_sid, StringType())
            spark.udf.register("extract_recording_sid", extract_recording_sid_udf)

            # Build the query using SafeQueryDSL
            query = (self.dsl
                        .select("*")
                        .from_table("csda", "tasksegments")
                        .where("DATE(agentConnectTimestamp) >= ?")
                        .add_param(self.start_date)
                        .where("taskId IN (SELECT taskId FROM task_ids_temp)")
                        .build())

            # Execute the query
            dfFinal = self.execute_safe_query(query[0], query[1])
            extract_recording_sid_udf = udf(self.extract_recording_sid, StringType())
            dfFinal = dfFinal.withColumn("RecordingSID", extract_recording_sid_udf(dfFinal["taskURL"]))
            dfFinal = dfFinal.toPandas()
            dfFinal = dfFinal[dfFinal['RecordingSID'] != '']

            print(f"Number of Records to be processed: {len(dfFinal)}")
            print("Start Download")

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_rsid = {executor.submit(self.download_recording_single, row['RecordingSID']): row['RecordingSID'] for _, row in dfFinal.iterrows()}
                results = []
                for future in tqdm(concurrent.futures.as_completed(future_to_rsid), total=len(future_to_rsid), desc="Downloading recordings"):
                    rsid = future_to_rsid[future]
                    try:
                        file_path = future.result()
                        results.append({'RecordingSID': rsid, 'file_path': file_path})
                    except Exception as exc:
                        self.logger.error(f'{rsid} generated an exception: {exc}')
                        results.append({'RecordingSID': rsid, 'file_path': 'Not Downloaded'})

            results_df = pd.DataFrame(results)
            dfFinal = dfFinal.merge(results_df, on='RecordingSID', how='left')

            print("Download Complete")

            intent_df = self.customer_metadata_extraction()

            if 'callSid' not in dfFinal.columns or 'callSid' not in intent_df.columns:
                raise ValueError("'callSid' column missing from one or both DataFrames")
            print("Merging Started")
            metadata = dfFinal.merge(intent_df, on=['callSid'], how='left')
            print("Merging Done")
            metadata.to_csv(self.calls_metadata_path, index=False)
            self.logger.info(f"Metadata saved to: {self.calls_metadata_path}")

            return metadata
        except Exception as e:
            self.logger.error(f"Error in recording_download_trigger: {str(e)}")
            raise

    def run(self) -> pd.DataFrame:
        """
        Execute the main workflow of the TwilioCallExtractor.

        This method orchestrates the entire extraction process by:
        1. Setting up the DataFrame.
        2. Triggering the recording download and processing.

        Returns:
            pd.DataFrame: The final metadata DataFrame after processing.

        Raises:
            Exception: If any error occurs during the execution.
        """
        try:
            self.setup_df()
            metadata = self.recording_download_trigger()
            self.logger.info("TwilioCallExtractor run completed successfully.")
            return metadata
        except Exception as e:
            self.logger.error(f"Error in TwilioCallExtractor run: {str(e)}")
            raise

    def validate_config(self) -> None:
        """
        Execute the main workflow of the TwilioCallExtractor.

        This method orchestrates the entire extraction process by:
        1. Setting up the DataFrame.
        2. Triggering the recording download and processing.

        Returns:
            pd.DataFrame: The final metadata DataFrame after processing.

        Raises:
            Exception: If any error occurs during the execution.
        """
        required_keys = ['spark_config', 'twilio', 'audio_directory', 
                         'call_sid_list_path', 'metadata_output_path', 
                         'num_of_records', 'max_workers', 'audio_response_path', 
                         'transfer_reason_path', 'intent_path','log_path']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        if 'account_sid' not in self.config['twilio'] or 'api_key' not in self.config['twilio'] or 'api_secret' not in self.config['twilio']:
            raise ValueError("Twilio configuration is incomplete")

    def safe_sql_identifier(self, identifier: str) -> str:
        """
        Sanitize and quote SQL identifiers to prevent SQL injection.

        Args:
            identifier (str): The SQL identifier to be sanitized.

        Returns:
            str: The sanitized and quoted SQL identifier.
        """
        return '`' + identifier.replace('`', '``') + '`'

    def execute_safe_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """
        Execute a SQL query safely using parameterization.

        Args:
            query (str): The SQL query to be executed.
            params (Dict[str, Any], optional): Parameters for the SQL query.

        Returns:
            Any: The result of the SQL query execution.

        Raises:
            Exception: If an error occurs during query execution.
        """
        try:
            if params:
                return spark.sql(query, params)
            else:
                return spark.sql(query)
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        extractor = TwilioCallExtractor("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/audio_file_retrieval_credentials.json")
        extractor.validate_config()
        metadata = extractor.run()
        print("Extraction completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

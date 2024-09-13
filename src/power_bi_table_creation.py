import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

class CallSidDataProcessor:
    """
    A class to process call SID-based data by merging various datasets, transforming
    columns, and converting the final DataFrame to a Spark DataFrame for further operations.

    Attributes:
        spark_session (SparkSession): Spark session to manage data operations.
    """

    def __init__(self, spark_session: SparkSession):
        """
        Initialize the CallSidDataProcessor with a Spark session.

        Args:
            spark_session (SparkSession): An instance of SparkSession.
        """
        self.spark_session = spark_session

    def define_spark_schema(self) -> StructType:
        """
        Define the schema for the Spark DataFrame.

        Returns:
            StructType: Schema definition for the DataFrame.
        """
        schema = StructType([
            StructField('bill_account_key', StringType(), True),
            StructField('file_path', StringType(), True),
            StructField('transcription', StringType(), True),
            StructField('summary', StringType(), True),
            StructField('topics', StringType(), True),
            StructField('summary_concise', StringType(), True),
            StructField('Category', StringType(), True),
            StructField('Main_Issue', StringType(), True),
            StructField('Steps_Taken', StringType(), True),
            StructField('Sentiment', StringType(), True),
            StructField('Urgency', StringType(), True),
            StructField('FollowUp_Actions', StringType(), True),
            StructField('RepeatedIssues', StringType(), True),
            StructField('CustomerLoyaltyIndicators', StringType(), True),
            StructField('TransferInvolved', StringType(), True),
            StructField('DepartmentTransferredTo', StringType(), True),
            StructField('IssueResolutionStatus', StringType(), True),
            StructField('SatisfactionScore', StringType(), True),
            StructField('ImprovementSuggestions', StringType(), True),
            StructField('taskId', StringType(), True),
            StructField('callSid', StringType(), True),
            StructField('callbackId', StringType(), True),
            StructField('workItemId', StringType(), True),
            StructField('taskSid', StringType(), True),
            StructField('segmentId', StringType(), True),
            StructField('segmentSid', StringType(), True),
            StructField('workGroupName', StringType(), True),
            StructField('workGroupQueueTimestamp', TimestampType(), True),
            StructField('skillGroupName', StringType(), True),
            StructField('queueName', StringType(), True),
            StructField('queueTime', IntegerType(), True),
            StructField('agentConnectTimestamp', TimestampType(), True),
            StructField('agentWrapupTimestamp', TimestampType(), True),
            StructField('endTimestamp', TimestampType(), True),
            StructField('inServiceLevel', StringType(), True),
            StructField('talkTime', IntegerType(), True),
            StructField('holdTime', IntegerType(), True),
            StructField('acwTime', IntegerType(), True),
            StructField('handleTime', IntegerType(), True),
            StructField('holdCount', IntegerType(), True),
            StructField('workerSid', StringType(), True),
            StructField('workerName', StringType(), True),
            StructField('workerCompany', StringType(), True),
            StructField('workerLocation', StringType(), True),
            StructField('workerManager', StringType(), True),
            StructField('taskType', StringType(), True),
            StructField('segmentConferenceYN', StringType(), True),
            StructField('segmentInterpreterYN', StringType(), True),
            StructField('segmentTransferredYN', StringType(), True),
            StructField('disconnectedBy', StringType(), True),
            StructField('callbackRequeuedYN', StringType(), True),
            StructField('taskURL', StringType(), True),
            StructField('insertDate', TimestampType(), True),
            StructField('RecordingSID', StringType(), True)
        ])
        return schema

    def read_data(self) -> dict:
        """
        Read and load data from multiple sources.

        Returns:
            dict: Dictionary containing the loaded data.
        """
        with open("/credentials/audio_file_retrieval_credentials.json", "r") as f:
            metadata_creds = json.load(f)

        with open("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/transcription.json", "r") as f:
            trans_creds = json.load(f)

        with open("/credentials/summary.json", "r") as f:
            summary_creds = json.load(f)

        with open("/credentials/topics.json", "r") as f:
            topic_creds = json.load(f)

        with open("/credentials/advanced_insights.json", "r") as f:
            adv_insights = json.load(f)

        metadata = pd.read_csv(metadata_creds['metadata_output_path'])
        transcription = pd.read_parquet(trans_creds['output_path'])
        summary = pd.read_parquet(summary_creds['output_path'])
        topics = pd.read_parquet(topic_creds['output_path'])
        advanced_insights = pd.read_csv(adv_insights['output_path'])

        return {
            "metadata": metadata,
            "transcription": transcription,
            "summary": summary,
            "topics": topics,
            "advanced_insights": advanced_insights
        }

    def process_data(self, data: dict) -> pd.DataFrame:
        """
        Process the loaded data by merging, concatenating, and transforming it into a final DataFrame.

        Args:
            data (dict): Dictionary containing the loaded data.

        Returns:
            pd.DataFrame: The processed final DataFrame.
        """
        summ_top = pd.concat([data['summary'], data['topics']], axis=1)
        summ_top = pd.concat([summ_top, data['transcription']], axis=1)[['file', 'transcription', 'summary', 'topics']]
        sum_top_trans = pd.concat([summ_top, data['advanced_insights']], axis=1)
        final = sum_top_trans.rename(columns={'file': 'file_path'}).merge(data['metadata'], on=['file_path'], how='inner')
        final.drop(columns=['User_Response', 'Intent', 'Intent_Score', 'IVR_Question'], axis=1, inplace=True)

        with open("/credentials/audio_file_retrieval_credentials.json", "r") as f:
            bill_account_creds = json.load(f)

        master_data = pd.read_parquet(bill_account_creds['call_sid_list_path'])

        front_end = final
        front_end = master_data.rename(columns={'Task_ID': 'taskId'})[['KY_BA', 'taskId']].merge(front_end, on=['taskId'],
                                                                                               how='inner').drop(
            columns=['Unnamed: 0.1', 'Unnamed: 0'])

        # Convert and clean datetime and integer columns
        datetime_columns = ['workGroupQueueTimestamp', 'agentConnectTimestamp', 'agentWrapupTimestamp', 'endTimestamp',
                            'insertDate']
        integer_columns = ['talkTime', 'holdTime', 'acwTime', 'queueTime', 'handleTime', 'holdCount']

        for col in datetime_columns:
            front_end[col] = pd.to_datetime(front_end[col], errors='coerce')

        for col in integer_columns:
            front_end[col] = front_end[col].fillna(0).astype(int)

        reordered_columns = [
            'KY_BA', 'file_path', 'transcription', 'summary', 'topics', 'Summary', 'Category', 'Main Issue',
            'Steps Taken', 'Sentiment', 'Urgency', 'Follow-Up Actions', 'Repeated Issues', 'Customer Loyalty Indicators',
            'Transfer Involved', 'Department Transferred To', 'Issue Resolution Status', 'Satisfaction Score',
            'Improvement Suggestions', 'taskId', 'callSid', 'callbackId', 'workItemId', 'taskSid', 'segmentId',
            'segmentSid', 'workGroupName', 'workGroupQueueTimestamp', 'skillGroupName', 'queueName', 'queueTime',
            'agentConnectTimestamp', 'agentWrapupTimestamp', 'endTimestamp', 'inServiceLevel', 'talkTime', 'holdTime',
            'acwTime', 'handleTime', 'holdCount', 'workerSid', 'workerName', 'workerCompany', 'workerLocation',
            'workerManager', 'taskType', 'segmentConferenceYN', 'segmentInterpreterYN', 'segmentTransferredYN',
            'disconnectedBy', 'callbackRequeuedYN', 'taskURL', 'insertDate', 'RecordingSID'
        ]

        front_end_reqd = front_end[reordered_columns]
        front_end_reqd = front_end_reqd.rename(columns={
            'Summary': 'summary_concise'
        })

        return front_end_reqd

    def convert_to_spark_df(self, pandas_df: pd.DataFrame) -> DataFrame:
        """
        Convert the processed pandas DataFrame to a Spark DataFrame.

        Args:
            pandas_df (pd.DataFrame): The pandas DataFrame to be converted.

        Returns:
            DataFrame: The resulting Spark DataFrame.
        """
        schema = self.define_spark_schema()
        spark_df = self.spark_session.createDataFrame(pandas_df, schema=schema)
        return spark_df

    def save_spark_df(self, spark_df: DataFrame, table_name: str, database_name: str = "default") -> None:
        """
        Save the Spark DataFrame to a table in the specified database.

        Args:
            spark_df (DataFrame): The Spark DataFrame to be saved.
            table_name (str): The name of the table to save the DataFrame.
            database_name (str): The name of the database to save the table. Defaults to "default".
        """
        spark_df.write.mode("overwrite").option("overwriteSchema", "True").saveAsTable(f"{database_name}.{table_name}")


def main():
    spark = SparkSession.builder.appName('DataConversion').getOrCreate()

    processor = CallSidDataProcessor(spark_session=spark)
    
    data = processor.read_data()
    processed_data = processor.process_data(data)
    processed_data.to_parquet(r'/data/final_output/final_output.parquets')
    final_spark_df = processor.convert_to_spark_df(processed_data)

    # Display the final Spark DataFrame
    final_spark_df.show()  # Using show instead of display for general spark dataframe in production

    # Save to table
    processor.save_spark_df(final_spark_df, table_name="call_sid_based_eval")

if __name__ == "__main__":
    main()

import os
import logging
from typing import Optional, Dict, Any
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, count, when
from pyspark.sql.types import StructType, StructField, StringType, DateType, TimestampType, IntegerType
import json
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
APPROVED_TABLES = {"csda.callHeader"}
APPROVED_COLUMNS = {
    "csda.callHeader": {
        "conversationId", "startTimestamp", "billAccountNumber", "KY_BA",
        "TASK_ID", "CALL_DATE", "CALL_START_TIME",
        "inIvrTime", "transferToAgentReason"
    }
}

class SparkJobException(Exception):
    """Custom exception for Spark job errors."""
    pass

def initialize_spark(app_name: str = "Production Spark Job") -> SparkSession:
    """
    Initialize Spark session with the specified app name.
    
    Args:
        app_name (str): The name of the Spark application.
    
    Returns:
        SparkSession: A configured Spark session object.
    """
    spark = (SparkSession.builder
             .appName(app_name)
             .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.shuffle.partitions", "auto")
             .getOrCreate())
    
    logger.info(f"Spark session initialized with app name: {app_name}")
    return spark

def configure_spark_storage(spark: SparkSession, config: Dict[str, str]) -> None:
    """
    Configures Spark to authenticate and access Azure storage account.
    
    Args:
        spark (SparkSession): The active Spark session.
        config (Dict[str, str]): Configuration dictionary containing storage details.
    """
    storage_account = config['storage_account']
    client_id = config['client_id']
    client_secret = config['client_secret']
    tenant_id = config['tenant_id']

    spark.conf.set("fs.azure.enable.check.access", "false")
    spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "OAuth")
    spark.conf.set(f"fs.azure.account.hns.enabled.{storage_account}.dfs.core.windows.net", "true")
    spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net", 
                   "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
    spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account}.dfs.core.windows.net", client_id)
    spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account}.dfs.core.windows.net", client_secret)
    spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account}.dfs.core.windows.net", 
                   f"https://login.microsoftonline.com/{tenant_id}/oauth2/token")
    
    logger.info(f"Spark storage configured for account: {storage_account}")

def get_tables(spark: SparkSession, database: str) -> DataFrame:
    """
    Retrieve all tables in the specified database.
    
    Args:
        spark (SparkSession): The active Spark session.
        database (str): The database name.
    
    Returns:
        DataFrame: A DataFrame containing the tables in the database.
    """
    try:
        tables = spark.sql(f"SHOW TABLES IN `{database}`")
        logger.info(f"Retrieved {tables.count()} tables from database: {database}")
        return tables
    except Exception as e:
        logger.error(f"Error retrieving tables from database {database}: {str(e)}")
        raise SparkJobException(f"Failed to retrieve tables: {str(e)}")

def get_agent_calls(spark: SparkSession, start_date: str, end_date: str) -> DataFrame:
    if "csda.callHeader" not in APPROVED_TABLES:
        raise SparkJobException("Unauthorized table access: csda.callHeader")

    try:
        query = """
        SELECT
            ch.conversationId,
            ch.startTimestamp,
            ch.billAccountNumber,
            ch.inIvrTime,
            ch.transferToAgentReason
        FROM csda.callHeader ch
        WHERE ch.inIvrTime > 2
            AND CAST(ch.startTimestamp AS date) BETWEEN ? AND ? 
            AND ch.transferToAgentReason IS NOT NULL
            AND ch.billAccountNumber IS NOT NULL
        """

        logger.info(f"Executing query: {query}")
        df = spark.sql(query, params=[start_date, end_date])
        
        logger.info(f"Query executed. DataFrame schema: {df.schema}")
        logger.info(f"Number of rows in DataFrame: {df.count()}")

        table_name = "csda.callHeader"
        columns = set(df.columns)
        
        logger.info(f"Columns in DataFrame: {columns}")
        logger.info(f"Approved columns: {APPROVED_COLUMNS.get(table_name)}")

        if not columns:
            logger.error("DataFrame has no columns")
            return None

        if not columns.issubset(APPROVED_COLUMNS.get(table_name)):
            unapproved_columns = columns - APPROVED_COLUMNS.get(table_name)
            logger.error(f"Columns {unapproved_columns} are not approved for table '{table_name}'.")
            return None
        
        result = df.select(
            col("conversationId").alias("TASK_ID"),
            col("startTimestamp").cast("date").alias("CALL_DATE"),
            col("startTimestamp").alias("CALL_START_TIME"),
            col("billAccountNumber").alias("KY_BA")
        )
        
        logger.info(f"Retrieved {result.count()} agent calls between {start_date} and {end_date}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving agent calls: {str(e)}")
        raise SparkJobException(f"Failed to retrieve agent calls: {str(e)}")

def get_frequent_callers(df: DataFrame) -> DataFrame:
    """
    Retrieve callers who made more than two distinct calls, including all fields.
    
    Args:
        df (DataFrame): The DataFrame containing agent calls data.
    
    Returns:
        DataFrame: A DataFrame containing frequent callers with all fields.
    """
    try:
        result = df.groupBy("KY_BA") \
                   .agg(count("TASK_ID").alias("NumberOfCalls"),
                        count(when(col("TASK_ID").isNotNull(), col("TASK_ID"))).alias("DistinctCalls")) \
                   .filter(col("DistinctCalls") > 2) \
                   .join(df, "KY_BA", "inner") \
                   .select("KY_BA", "TASK_ID", "CALL_DATE", "CALL_START_TIME", "NumberOfCalls")
        
        logger.info(f"Identified {result.count()} frequent callers")
        return result
    except Exception as e:
        logger.error(f"Error identifying frequent callers: {str(e)}")
        raise SparkJobException(f"Failed to identify frequent callers: {str(e)}")

def validate_date(date_string: str) -> bool:
    """
    Validate the date string format.
    
    Args:
        date_string (str): The date string to validate.
    
    Returns:
        bool: True if the date string is valid, False otherwise.
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the configuration JSON file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise SparkJobException(f"Failed to load configuration: {str(e)}")

def main():
    """
    Main function to run the production Spark job.
    
    Args:
        config_path (str): Path to the configuration JSON file.
    """
    try:
        # Load configuration
        config = load_config("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/bill_account_secrets.json")
        
        # Initialize Spark session
        spark = initialize_spark(config.get('app_name', 'Production Spark Job'))

        # Configure Spark storage
        configure_spark_storage(spark, config)

        # Validate date inputs
        start_date = config['start_date']
        end_date = config['end_date']
        if not (validate_date(start_date) and validate_date(end_date)):
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")

        # Retrieve tables and agent call data
        df_tables = get_tables(spark, config['database'])
        df_tables.show()

        df_agent_calls = get_agent_calls(spark, start_date, end_date)
        df_frequent_callers = get_frequent_callers(df_agent_calls)
        
        df_frequent_callers.show()

        # Specify the output path
        output_path = config['output_path']
        
        # Store the DataFrame with all fields to the specified location
        df_frequent_callers.toPandas().to_parquet(output_path) #.write.mode("overwrite").parquet(output_path)
        logger.info(f"DataFrame successfully saved to {output_path}")

    except SparkJobException as e:
        logger.error(f"Spark job failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

import os
import logging
from typing import Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_spark(app_name: str = "Optimize Production Code") -> SparkSession:
    """
    Initialize Spark session with the specified app name.
    
    Args:
        app_name (str): The name of the Spark application.
    
    Returns:
        SparkSession: A Spark session object.
    """
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark

def configure_spark_storage(spark: SparkSession, storage_account: str, client_id: str, client_secret: str, tenant_id: str) -> None:
    """
    Configures Spark to authenticate and access Azure storage account.
    
    Args:
        spark (SparkSession): The active Spark session.
        storage_account (str): The Azure storage account name.
        client_id (str): The service principal client ID.
        client_secret (str): The service principal client secret.
        tenant_id (str): The Azure tenant ID.
    """
    spark.conf.set("fs.azure.enable.check.access", "false")
    spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "OAuth")
    spark.conf.set(f"fs.azure.account.hns.enabled.{storage_account}.dfs.core.windows.net", "true")
    spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net", 
                   "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
    spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account}.dfs.core.windows.net", client_id)
    spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account}.dfs.core.windows.net", client_secret)
    spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account}.dfs.core.windows.net", 
                   f"https://login.microsoftonline.com/{tenant_id}/oauth2/token")

def get_tables(spark: SparkSession, database: str) -> DataFrame:
    """
    Retrieve all tables in the specified database.
    
    Args:
        spark (SparkSession): The active Spark session.
        database (str): The database name.
    
    Returns:
        DataFrame: A DataFrame containing the tables in the database.
    """
    query = f"SHOW TABLES IN {database}"
    return spark.sql(query)

def get_agent_calls(spark: SparkSession, start_date: str, end_date: str) -> DataFrame:
    """
    Retrieve agent calls data between the given start and end date.
    
    Args:
        spark (SparkSession): The active Spark session.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
    
    Returns:
        DataFrame: A DataFrame containing the agent calls information.
    """
    query = f"""
    SELECT
        ch.conversationId AS TASK_ID,
        CAST(ch.startTimestamp AS date) AS CALL_DATE,
        ch.startTimestamp AS CALL_START_TIME,
        ch.billAccountNumber AS KY_BA
    FROM csda.callHeader ch
    WHERE ch.inIvrTime > 2
        AND CAST(ch.startTimestamp AS date) BETWEEN '{start_date}' AND '{end_date}'
        AND transferToAgentReason IS NOT NULL
        AND ch.billAccountNumber IS NOT NULL
    """
    return spark.sql(query)

def get_frequent_callers(df: DataFrame) -> DataFrame:
    """
    Retrieve callers who made more than two distinct calls, including all fields.
    
    Args:
        df (DataFrame): The DataFrame containing agent calls data.
    
    Returns:
        DataFrame: A DataFrame containing frequent callers with all fields.
    """
    df.createOrReplaceTempView("dfAgentCalls")
    
    query = """
    WITH FrequentCallers AS (
        SELECT 
            KY_BA,
            COUNT(DISTINCT TASK_ID) AS NumberOfCalls
        FROM dfAgentCalls
        GROUP BY KY_BA
        HAVING COUNT(DISTINCT TASK_ID) > 2
    )
    SELECT 
        ac.KY_BA,
        ac.TASK_ID,
        ac.CALL_DATE,
        ac.CALL_START_TIME,
        fc.NumberOfCalls
    FROM dfAgentCalls ac
    INNER JOIN FrequentCallers fc ON ac.KY_BA = fc.KY_BA
    ORDER BY fc.NumberOfCalls DESC, ac.KY_BA, ac.CALL_DATE, ac.CALL_START_TIME
    """
    return spark.sql(query)

def main():
    """
    Main function to run the optimized Spark code.
    """
    # Initialize Spark session
    spark = initialize_spark()

    # Secrets and configuration parameters
    adanexpr_dbricks_id = dbutils.secrets.get(scope="pplz-key-adanexpr", key="Azure-SP-ADANEXPR-DBricks-ID")
    adanexpr_dbricks_pwd = dbutils.secrets.get(scope="pplz-key-adanexpr", key="Azure-SP-ADANEXPR-DBricks-PWD")
    ppl_tenant_id = dbutils.secrets.get(scope="pplz-key-adanexpr", key="tenant-id-adanexpr")

    dpcore_storage_acct = dbutils.secrets.get(scope="pplz-key-adanexpr", key="storage-account-dpcore")
    dpcore_dbricks_id = dbutils.secrets.get(scope="pplz-key-adanexpr", key="Azure-SP-ADANEXPR-DPCore-DBricks-ID")
    dpcore_dbricks_pwd = dbutils.secrets.get(scope="pplz-key-adanexpr", key="Azure-SP-ADANEXPR-DPCore-DBricks-PWD")

    # Configure Spark storage
    configure_spark_storage(spark, dpcore_storage_acct, dpcore_dbricks_id, dpcore_dbricks_pwd, ppl_tenant_id)

    with open("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/bill_account_secrets.json", "r") as f:
        creds = json.load(f)

    # Parameters
    start_date = creds['start_date']
    end_date = creds['end_date']

    # Retrieve tables and agent call data
    df_tables = get_tables(spark, "csda")
    df_tables.show()

    df_agent_calls = get_agent_calls(spark, start_date, end_date)
    df_agent_calls.show()

    df_frequent_callers = get_frequent_callers(df_agent_calls)
    output_path = creds['output_path']
    df_frequent_callers_pd = df_frequent_callers.toPandas().reset_index(drop=True).iloc[:100, :]
    print(len(df_frequent_callers_pd))
    df_frequent_callers_pd.to_parquet(output_path)
    # Store the DataFrame with all fields to the specified location
    #df_frequent_callers.write.mode("overwrite").parquet(output_path)
    logger.info(f"DataFrame successfully saved to {output_path}")

if __name__ == "__main__":
    main()

################################
### 1. Importaciones
################################
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, count, countDistinct, lit, month, quarter
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import weekofyear, year
from pyspark.sql.types import *


################################################
### 2. Configuración de rutas y carga de datos
################################################
def create_spark_session():
    spark = SparkSession.builder.appName("ModeloEstrellaRestaurante").getOrCreate()
    return spark


def read_data(spark, input_data):
    df_categories = spark.read.csv(f"{input_data}/CATEGORIES.csv", header=True, inferSchema=True)
    df_customers = spark.read.csv(f"{input_data}/CUSTOMERS.csv", header=True, inferSchema=True)
    df_employees = spark.read.csv(f"{input_data}/EMPLOYEES.csv", header=True, inferSchema=True)
    df_orders = spark.read.csv(f"{input_data}/ORDERS.csv", header=True, inferSchema=True)
    df_prices = spark.read.csv(f"{input_data}/PRICES.csv", header=True, inferSchema=True)
    df_products = spark.read.csv(f"{input_data}/PRODUCTS.csv", header=True, inferSchema=True)
    df_sizes = spark.read.csv(f"{input_data}/SIZES.csv", header=True, inferSchema=True)
    return df_categories, df_customers, df_employees, df_orders, df_prices, df_products, df_sizes


# Configuración de rutas de entrada y salida
input_data = "gs://test-cloud-storage-nahum/input"
output_data = "gs://test-cloud-storage-nahum/output"

# Inicializar sesión de Spark
spark = create_spark_session()

# Leer datos
df_categories, df_customers, df_employees, df_orders, df_prices, df_products, df_sizes = read_data(spark, input_data)


#####################################################
### 3. Funciones para crear tablas de dimensiones
#####################################################
def create_dim_employees(df_employees):
    return df_employees.select(
        col("EMPLOYEE_ID").alias("Employee_ID"),
        concat(col("FNAMEE"), lit(" "), col("LNAMEE")).alias("employee_FullName"),
    ).distinct()


def create_dim_customers(df_customers):
    return df_customers.select(
        col("CUSTOMER_ID").alias("CUSTOMER_ID"),
        concat(col("FNAMEC"), lit(" "), col("LNAMEC")).alias("CUSTOMER_FULLNAME"),
    ).distinct()


def create_dim_products(df_products, df_categories, df_prices, df_sizes):
    return (
        df_products.join(df_categories.withColumnRenamed("DESCRIPTION", "PRODUCT_CATEGORY"), "CATEGORY_ID", "left")
        .join(df_prices.withColumnRenamed("DESCRIPTION", "PRODUCT_PRICE"), "PRICE_ID", "left")
        .join(df_sizes.withColumnRenamed("DESCRIPTION", "PRODUCT_SIZE"), "SIZE_ID", "left")
        .select(
            df_products["PRODUCT_ID"].alias("PRODUCT_ID"),
            df_products["NAMEP"].alias("PRODUCT_NAME"),
            col("PRODUCT_CATEGORY"),
            col("PRODUCT_PRICE"),
            col("PRODUCT_SIZE"),
        )
        .distinct()
    )


def create_dim_time(df_orders):
    return df_orders.select(
        col("ORDER_DATE").alias("TIME_ID"),
        month(col("ORDER_DATE")).alias("MONTHO"),
        quarter(col("ORDER_DATE")).alias("trimester"),
        year(col("ORDER_DATE")).alias("semester"),
        weekofyear(col("ORDER_DATE")).alias("week_day"),
    ).distinct()


# Crear tablas de dimensiones
dim_employees = create_dim_employees(df_employees)
dim_customers = create_dim_customers(df_customers)
dim_products = create_dim_products(df_products, df_categories, df_prices, df_sizes)
dim_time = create_dim_time(df_orders)


##############################################
### 4. Función para crear tabla de hechos
##############################################
def create_hechos_orders_sales(df_orders, df_products, df_prices):
    return (
        df_orders.join(df_products, "PRODUCT_ID", "left")
        .join(df_prices, "PRICE_ID", "left")
        .withColumn("TOTAL_AMOUNT", col("AMOUNT") * col("DESCRIPTION").cast("float"))
        .select(
            col("ORDER_ID").alias("HECHO_ID"),
            col("EMPLOYEE_ID"),
            col("CUSTOMER_ID"),
            col("PRODUCT_ID"),
            col("ORDER_DATE").alias("TIME_ID"),
            col("AMOUNT").alias("QUANTITY_SALE_PRODUCTS"),
            "TOTAL_AMOUNT",
        )
    )


# Crear tabla de hechos
hechos_orders_sales = create_hechos_orders_sales(df_orders, df_products, df_prices)


# Función para guardar DataFrames en formato Parquet
def save_parquet(df, output_path, partition_by=None):
    if partition_by:
        df.write.mode("overwrite").partitionBy(partition_by).parquet(output_path)
    else:
        df.write.mode("overwrite").parquet(output_path)


# Guardar tablas de dimensiones y hechos
save_parquet(dim_employees, os.path.join(output_data, "Dim_Employees"))
save_parquet(dim_customers, os.path.join(output_data, "Dim_Customers"))
save_parquet(dim_products, os.path.join(output_data, "Dim_Products"))
save_parquet(dim_time, os.path.join(output_data, "Dim_TIME"))
save_parquet(hechos_orders_sales, os.path.join(output_data, "Hechos_Orders_Sales"))


################################
### 5. Preguntas de Negocio
################################
# 1. Función para calcular el monto total de ventas por empleado
def ventas_por_empleado(hechos_orders_sales, dim_employees):
    """
    Calcula el monto total de ventas por empleado.
    """
    return (
        hechos_orders_sales.join(dim_employees, hechos_orders_sales["EMPLOYEE_ID"] == dim_employees["Employee_ID"])
        .groupBy(dim_employees["Employee_ID"], "employee_FullName")
        .agg(spark_sum("TOTAL_AMOUNT").alias("monto_total_ventas"))
        .orderBy("monto_total_ventas", ascending=False)
    )


# 2. Función para analizar ventas por mes, trimestre y semestre
def ventas_por_tiempo(hechos_orders_sales, dim_time):
    """
    Analiza el monto total de ventas por mes, trimestre y semestre.
    """
    return (
        hechos_orders_sales.join(dim_time, hechos_orders_sales["TIME_ID"] == dim_time["TIME_ID"])
        .groupBy("MONTHO", "trimester", "semester")
        .agg(spark_sum("TOTAL_AMOUNT").alias("total_ventas"))
        .orderBy("semester", "trimester", "MONTHO")
    )


# 3. Función para encontrar el día de la semana con mayor consumo
def dia_semana_mayor_consumo(hechos_orders_sales, dim_time):
    """
    Encuentra el día de la semana con el mayor número de ventas.
    """
    return (
        hechos_orders_sales.join(dim_time, hechos_orders_sales["TIME_ID"] == dim_time["TIME_ID"])
        .groupBy("week_day")
        .agg(count("*").alias("conteo_ventas"))
        .orderBy("conteo_ventas", ascending=False)
        .limit(1)
    )


# 4. Función para calcular la cantidad de clientes atendidos por mes
def cantidad_clientes_por_mes(hechos_orders_sales, dim_time):
    """
    Calcula la cantidad de clientes únicos atendidos por mes.
    """
    return (
        hechos_orders_sales.join(dim_time, hechos_orders_sales["TIME_ID"] == dim_time["TIME_ID"])
        .groupBy("MONTHO")
        .agg(countDistinct("CUSTOMER_ID").alias("cantidad_clientes"))
        .orderBy("MONTHO")
    )


# 5. Función para encontrar los productos más y menos vendidos
def productos_mas_y_menos_vendidos(hechos_orders_sales, dim_products):
    """
    Calcula los productos más y menos vendidos.
    """
    productos_venta = (
        hechos_orders_sales.join(dim_products, hechos_orders_sales["PRODUCT_ID"] == dim_products["PRODUCT_ID"])
        .groupBy(dim_products["PRODUCT_ID"], "PRODUCT_NAME", "PRODUCT_CATEGORY", "PRODUCT_PRICE")
        .agg(spark_sum("QUANTITY_SALE_PRODUCTS").alias("total_vendido"))
    )

    # Producto más vendido
    producto_mas_vendido = (
        productos_venta.orderBy(col("total_vendido").desc()).limit(1).withColumn("demanda", lit("Mas Vendido"))
    )

    # Producto menos vendido
    producto_menos_vendido = (
        productos_venta.orderBy(col("total_vendido").asc()).limit(1).withColumn("demanda", lit("Menos Vendido"))
    )

    # Unir los resultados
    return producto_mas_vendido.union(producto_menos_vendido)


# Ejecutar las funciones y mostrar los resultados
ventas_empleado_df = ventas_por_empleado(hechos_orders_sales, dim_employees)
ventas_empleado_df.show()

ventas_tiempo_df = ventas_por_tiempo(hechos_orders_sales, dim_time)
ventas_tiempo_df.show()

dia_mayor_consumo_df = dia_semana_mayor_consumo(hechos_orders_sales, dim_time)
dia_mayor_consumo_df.show()

clientes_mes_df = cantidad_clientes_por_mes(hechos_orders_sales, dim_time)
clientes_mes_df.show()

productos_ventas_df = productos_mas_y_menos_vendidos(hechos_orders_sales, dim_products)
productos_ventas_df.show()


################################
### 6. Métricas
################################
# 1. Función para calcular el monto total de una orden
def monto_total_por_orden(hechos_orders_sales, dim_products):
    """
    Calcula el monto total de cada orden (precio * cantidad).
    """
    return (
        hechos_orders_sales.join(dim_products, hechos_orders_sales["PRODUCT_ID"] == dim_products["PRODUCT_ID"])
        .withColumn("monto_total", col("PRODUCT_PRICE") * col("QUANTITY_SALE_PRODUCTS"))
        .select(
            hechos_orders_sales["HECHO_ID"],
            dim_products["PRODUCT_ID"],
            "QUANTITY_SALE_PRODUCTS",
            "PRODUCT_PRICE",
            "monto_total",
        )
    )


# 2. Función para contar la cantidad de clientes atendidos durante el mes
def cantidad_clientes_atendidos_mes(hechos_orders_sales, dim_time):
    """
    Cuenta los clientes únicos atendidos durante cada mes.
    """
    return (
        hechos_orders_sales.join(dim_time, hechos_orders_sales["TIME_ID"] == dim_time["TIME_ID"])
        .groupBy("MONTHO")
        .agg(countDistinct("CUSTOMER_ID").alias("cantidad_clientes"))
        .orderBy("MONTHO")
    )


monto_total_orden_df = monto_total_por_orden(hechos_orders_sales, dim_products)
monto_total_orden_df.show()

clientes_atendidos_mes_df = cantidad_clientes_atendidos_mes(hechos_orders_sales, dim_time)
clientes_atendidos_mes_df.show()


# 3. Finalizar sesión de Spark
spark.stop()


# 03_gold_analytics.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# =========================
# Configuración
# =========================
spark = SparkSession.builder.getOrCreate()

CATALOG = "airline_mantenimiento"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}"
)

# =========================
# Función para guardar Gold
# =========================
def guardar(df, nombre):
    df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.{nombre}")

    print(f"Tabla creada: {nombre} - {df.count()} registros")


# =========================
# 1. Leer Silver
# =========================
vuelos_slv = spark.table(
    f"{CATALOG}.{SILVER_SCHEMA}.vuelos"
)

aeronaves_slv = spark.table(
    f"{CATALOG}.{SILVER_SCHEMA}.aeronaves"
)

aeropuertos_slv = spark.table(
    f"{CATALOG}.{SILVER_SCHEMA}.aeropuertos"
)

mantenimientos_slv = spark.table(
    f"{CATALOG}.{SILVER_SCHEMA}.mantenimientos"
)


# =========================
# 2. Vuelos enriquecidos
# =========================
vuelos_gold = (
    vuelos_slv.alias("v")
    .join(
        aeronaves_slv.alias("a"),
        F.col("v.aeronave_id") == F.col("a.aeronave_id"),
        "left"
    )
    .join(
        aeropuertos_slv.alias("o"),
        F.col("v.origen_id") == F.col("o.aeropuerto_id"),
        "left"
    )
    .join(
        aeropuertos_slv.alias("d"),
        F.col("v.destino_id") == F.col("d.aeropuerto_id"),
        "left"
    )
    .select(
        F.col("v.vuelo_id").alias("vuelo_id"),
        F.col("v.fecha").alias("fecha"),
        F.trunc(F.col("v.fecha"), "month").alias("mes"),
        F.col("v.aeronave_id").alias("aeronave_id"),
        F.col("a.modelo").alias("modelo"),
        F.col("a.fabricante").alias("fabricante"),
        F.col("a.anio_fabricacion").alias("anio_fabricacion"),
        (
            F.year(F.col("v.fecha")) -
            F.col("a.anio_fabricacion")
        ).alias("edad_aeronave"),
        F.col("v.origen_id").alias("origen_id"),
        F.col("o.nombre").alias("origen_nombre"),
        F.col("o.ciudad").alias("origen_ciudad"),
        F.col("o.pais").alias("origen_pais"),
        F.col("v.destino_id").alias("destino_id"),
        F.col("d.nombre").alias("destino_nombre"),
        F.col("d.ciudad").alias("destino_ciudad"),
        F.col("d.pais").alias("destino_pais"),
        F.col("v.estado").alias("estado"),
        F.col("v.duracion_min").alias("duracion_min"),
        (
            F.col("v.estado") != F.lit("cancelado")
        ).alias("operado")
    )
)

assert vuelos_gold.count() == vuelos_slv.count(), \
    "La union modifico la cantidad de vuelos"

guardar(vuelos_gold, "vuelos_enriquecidos")


# =========================
# 3. Mantenimientos enriquecidos
# =========================
mantenimientos_gold = (
    mantenimientos_slv.alias("m")
    .join(
        aeronaves_slv.alias("a"),
        F.col("m.aeronave_id") == F.col("a.aeronave_id"),
        "left"
    )
    .select(
        F.col("m.mantenimiento_id").alias("mantenimiento_id"),
        F.col("m.fecha").alias("fecha"),
        F.trunc(F.col("m.fecha"), "month").alias("mes"),
        F.col("m.aeronave_id").alias("aeronave_id"),
        F.col("a.modelo").alias("modelo"),
        F.col("a.fabricante").alias("fabricante"),
        F.col("a.anio_fabricacion").alias("anio_fabricacion"),
        (
            F.year(F.col("m.fecha")) -
            F.col("a.anio_fabricacion")
        ).alias("edad_aeronave"),
        F.col("m.tipo").alias("tipo"),
        F.col("m.costo_usd").alias("costo_usd"),
        F.col("m.duracion_hr").alias("duracion_hr")
    )
)

assert mantenimientos_gold.count() == mantenimientos_slv.count(), \
    "La union modifico los mantenimientos"

guardar(
    mantenimientos_gold,
    "mantenimientos_enriquecidos"
)


# =========================
# 4. KPI de puntualidad
# =========================
puntualidad = (
    vuelos_gold
    .groupBy("modelo", "fabricante", "origen_pais", "mes")
    .agg(
        F.count("*").alias("total_vuelos"),

        F.sum(
            F.when(F.col("operado"), 1).otherwise(0)
        ).alias("vuelos_operados"),

        F.sum(
            F.when(F.col("estado") == "a_tiempo", 1)
            .otherwise(0)
        ).alias("vuelos_a_tiempo"),

        F.sum(
            F.when(F.col("estado") == "retrasado", 1)
            .otherwise(0)
        ).alias("vuelos_retrasados"),

        F.sum(
            F.when(F.col("estado") == "cancelado", 1)
            .otherwise(0)
        ).alias("vuelos_cancelados"),

        F.round(
            F.avg(
                F.when(
                    F.col("operado"),
                    F.col("duracion_min")
                )
            ), 2
        ).alias("duracion_promedio_operados_min")
    )
    .withColumn(
        "puntualidad_operados_pct",
        F.round(
            F.when(
                F.col("vuelos_operados") > 0,
                F.col("vuelos_a_tiempo") * 100.0 /
                F.col("vuelos_operados")
            ), 2
        )
    )
    .withColumn(
        "cancelacion_pct",
        F.round(
            F.col("vuelos_cancelados") * 100.0 /
            F.col("total_vuelos"), 2
        )
    )
)

guardar(puntualidad, "kpi_puntualidad_vuelos")


# =========================
# 5. KPI costos de mantenimiento
# =========================
costos = (
    mantenimientos_gold
    .groupBy(
        "aeronave_id",
        "modelo",
        "fabricante",
        "tipo",
        "mes"
    )
    .agg(
        F.count("*").alias("total_mantenimientos"),
        F.sum("costo_usd").alias("costo_total_usd"),
        F.round(
            F.avg("costo_usd"), 2
        ).alias("costo_promedio_usd"),
        F.sum("duracion_hr").alias("duracion_total_hr"),
        F.round(
            F.avg("duracion_hr"), 2
        ).alias("duracion_promedio_hr")
    )
)

guardar(costos, "kpi_costo_mantenimiento")


# =========================
# 6. Resumen mensual por aeronave
# =========================
actividad = (
    vuelos_gold
    .groupBy("aeronave_id", "mes")
    .agg(
        F.count("*").alias("total_vuelos"),
        F.sum(
            F.when(F.col("operado"), 1).otherwise(0)
        ).alias("vuelos_operados")
    )
)

mantenimiento_mes = (
    mantenimientos_gold
    .groupBy("aeronave_id", "mes")
    .agg(
        F.count("*").alias("total_mantenimientos"),
        F.sum("costo_usd").alias("costo_total_usd")
    )
)

resumen = (
    actividad
    .join(
        mantenimiento_mes,
        ["aeronave_id", "mes"],
        "full_outer"
    )
    .fillna({
        "total_vuelos": 0,
        "vuelos_operados": 0,
        "total_mantenimientos": 0,
        "costo_total_usd": 0.0
    })
)

guardar(resumen, "resumen_aeronave_mes")


# =========================
# 7. Base para Machine Learning
# =========================
base_ml = mantenimientos_gold.select(
    "mantenimiento_id",
    "fecha",
    "aeronave_id",
    "tipo",
    "modelo",
    "fabricante",
    "edad_aeronave",
    "costo_usd"
)

guardar(base_ml, "base_ml_mantenimiento")


# =========================
# 8. Validaciones finales
# =========================
assert vuelos_gold.count() == vuelos_slv.count()

assert mantenimientos_gold.count() == mantenimientos_slv.count()

assert base_ml.count() == mantenimientos_slv.count()

assert puntualidad.agg(
    F.sum("total_vuelos")
).first()[0] == vuelos_slv.count()

assert costos.agg(
    F.sum("total_mantenimientos")
).first()[0] == mantenimientos_slv.count()

costo_origen = mantenimientos_slv.agg(
    F.sum("costo_usd")
).first()[0]

costo_gold = costos.agg(
    F.sum("costo_total_usd")
).first()[0]

assert abs(costo_origen - costo_gold) < 0.01, \
    "Los costos no coinciden"

print("VALIDACIONES GOLD CORRECTAS")
print("Total vuelos:", vuelos_gold.count())
print("Total mantenimientos:", mantenimientos_gold.count())
print("Costo total USD:", round(costo_gold, 2))
print("Capa Gold finalizada correctamente")

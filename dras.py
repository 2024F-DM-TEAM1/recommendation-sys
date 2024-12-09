import os
import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import col, udf, split, when, lit
from pyspark.sql.types import FloatType
import numpy as np
import shutil


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_terminal_size():
    columns, _ = shutil.get_terminal_size()
    return columns


def center_text(text, width=None):
    if width is None:
        width = get_terminal_size()
    return text.center(width)


def display_title():
    title_art = """
    ██████╗ ██████╗  █████╗ ███████╗
    ██╔══██╗██╔══██╗██╔══██╗██╔════╝
    ██║  ██║██████╔╝███████║███████╗
    ██║  ██║██╔══██╗██╔══██║╚════██║
    ██████╔╝██║  ██║██║  ██║███████║
    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
    """

    subtitle = """
┌───────────────────────────────────────────────────────────────────────────────┐
│   Developer Recruitment Analysis & Recommendation System v1.0                 │
│                                                                               │
│      - [A] Analysis Mode    : Tech stack association rules                    │
│      - [R] Recommendation   : Recruitment recommendation based on user input  │
└───────────────────────────────────────────────────────────────────────────────┘
    """

    decoration = """
    ⚡️ Powered by MSLee, MJSeo, TYYoon ⚡️
    """

    term_width = get_terminal_size()

    print("\n")
    for line in title_art.split('\n'):
        print(center_text(line, term_width))

    for line in subtitle.split('\n'):
        print(center_text(line, term_width))

    for line in decoration.split('\n'):
        print(center_text(line, term_width))


def analysis_mode():
    clear_screen()
    print("Analysis Results\n")

    print(f"{'Antecedent':<30} {'Consequent':<25} {'Support':<13} {'Confidence':<13} {'Lift':<13}")
    print("-" * 100)

    results = [
        ("AI", "머신러닝", 0.05, 0.70, 4.38),
        ("DeepLearning", "머신러닝", 0.04, 0.93, 5.85),
        ("JavaScript", "프론트엔드", 0.05, 0.38, 3.70),
        ("PyTorch", "머신러닝", 0.05, 0.81, 5.10),
        ("React", "프론트엔드", 0.04, 0.70, 6.80),
        ("인공지능", "머신러닝", 0.10, 0.83, 5.19),
        ("Python, AI", "머신러닝", 0.04, 0.86, 5.40),
        ("AI", "머신러닝, 인공지능", 0.05, 0.70, 7.17),
        ("AI, 인공지능", "머신러닝", 0.05, 0.70, 4.38),
        ("Python, C", "머신러닝", 0.04, 0.50, 3.15),
        ("C, 인공지능", "머신러닝", 0.05, 0.84, 5.30),
        ("Python, C++", "머신러닝", 0.06, 0.49, 3.07),
        ("C++, 인공지능", "머신러닝", 0.06, 0.87, 5.48),
        ("DeepLearning", "머신러닝, 인공지능", 0.04, 0.93, 9.57),
        ("DeepLearning, 인공지능", "머신러닝", 0.04, 0.93, 5.85),
        ("Embedded", "HW, 임베디드", 0.06, 0.46, 3.84),
        ("JavaScript", "개발자, 서버", 0.04, 0.29, 2.63),
        ("MySQL", "백엔드, 서버", 0.05, 0.65, 4.04),
        ("Node.js", "백엔드, 서버", 0.04, 0.68, 4.21),
        ("PyTorch", "머신러닝, 인공지능", 0.04, 0.62, 6.38),
        ("인공지능, PyTorch", "머신러닝", 0.04, 1.00, 6.30),
        ("Python", "머신러닝, 인공지능", 0.07, 0.27, 2.75),
        ("Python, 인공지능", "머신러닝", 0.07, 0.92, 5.81),
        ("React", "개발자, 프론트엔드", 0.04, 0.60, 10.20),
        ("개발자, React", "프론트엔드", 0.04, 0.71, 6.86),
        ("백엔드", "개발자, 서버", 0.09, 0.55, 5.01),
        ("Python, AI", "머신러닝, 인공지능", 0.04, 0.86, 8.83),
        ("Python, AI, 인공지능", "머신러닝", 0.04, 0.86, 5.40),
        ("Python, C, C++", "머신러닝", 0.04, 0.55, 3.43),
        ("C, C++", "머신러닝, 인공지능", 0.05, 0.26, 2.66),
        ("C, C++, 인공지능", "머신러닝", 0.05, 0.84, 5.30),
        ("C, Embedded", "SW, 솔루션", 0.06, 0.63, 2.80),
        ("Python, C", "머신러닝, 인공지능", 0.04, 0.50, 5.15),
        ("Python, C, 인공지능", "머신러닝", 0.04, 0.86, 5.40),
        ("C++, C#", "SW, 솔루션", 0.05, 0.61, 2.68),
        ("Embedded, C++", "SW, 솔루션", 0.06, 0.73, 3.23),
        ("Python, C++", "머신러닝, 인공지능", 0.04, 0.38, 3.96),
        ("Python, C++, 인공지능", "머신러닝", 0.04, 0.88, 5.56),
        ("Linux, Embedded", "SW, 솔루션", 0.04, 0.61, 2.69),
        ("C, C++, Embedded", "SW, 솔루션", 0.05, 0.72, 3.18),
        ("Python, C, C++", "머신러닝, 인공지능", 0.04, 0.55, 5.62),
        ("Python, C, C++, 인공지능", "머신러닝", 0.04, 0.86, 5.40),
        ("C, Linux, Embedded", "SW, 솔루션", 0.04, 0.68, 3.02),
        ("Linux, Embedded, C++", "SW, 솔루션", 0.04, 0.75, 3.31),
        ("C, Linux, Embedded, C++", "SW, 솔루션", 0.04, 0.75, 3.31)
    ]

    for ant, cons, sup, conf, lift in results:
        print(f"{ant:<30} {cons:<25} {sup:<13.2f} {conf:<13.2f} {lift:<13.2f}")

    print("\nPress Enter to return to main menu...")
    input()


def recommendation_mode():
    clear_screen()

    spark = SparkSession.builder \
        .appName("Job Recommendation System") \
        .getOrCreate()

    job_postings = spark.read.csv("./data/processed_data.csv", header=True)

    job_postings = job_postings.select("comp_name", "category", "tech_stack")

    job_postings = job_postings.withColumn(
        "tech_stack_array",
        split(col("tech_stack"), r"\|")
    )

    job_postings = job_postings.withColumn(
        "tech_stack_array",
        when(col("tech_stack_array").isNull(), lit([])).otherwise(col("tech_stack_array"))
    )

    vectorizer = CountVectorizer(inputCol="tech_stack_array", outputCol="features")
    vectorizer_model = vectorizer.fit(job_postings)
    job_postings_vectorized = vectorizer_model.transform(job_postings)

    user_competency = input("Describe your competence (within 500 characters): ")
    print("-" * 70)
    print("-" * 70)
    user_input = input("Enter your technology stack, separated by commas(,): ")
    user_tech_stack = [tech.strip().lower().capitalize() for tech in user_input.split(",")]
    user_df = spark.createDataFrame([(0, user_tech_stack)], ["id", "tech_stack_array"])
    user_vectorized = vectorizer_model.transform(user_df)

    user_features = user_vectorized.select("features").collect()[0]["features"]

    def cosine_similarity(v1, v2):
        v1_array = np.array(v1.toArray())
        v2_array = np.array(v2.toArray())
        dot_product = np.dot(v1_array, v2_array)
        norm_v1 = np.linalg.norm(v1_array)
        norm_v2 = np.linalg.norm(v2_array)
        return float(dot_product / (norm_v1 * norm_v2) if norm_v1 and norm_v2 else 0.0)

    cosine_similarity_udf = udf(lambda x: cosine_similarity(x, user_features), FloatType())

    job_postings_with_similarity = job_postings_vectorized.withColumn(
        "similarity",
        cosine_similarity_udf(col("features"))
    )

    top_recommendations = job_postings_with_similarity.orderBy(
        col("similarity").desc()
    ).limit(20)

    top_recommendations.select("comp_name", "category", "similarity").show()


def main():
    while True:
        clear_screen()
        display_title()
        mode = input(center_text("\nEnter mode (A: Analysis, R: Recommendation, Q: Quit): ")).upper()

        if mode == 'A':
            analysis_mode()
        elif mode == 'R':
            recommendation_mode()
            input("\nPress Enter to continue...")
        elif mode == 'Q':
            print(center_text("\nGoodbye! Thank you for using DRAS."))
            break
        else:
            print(center_text("\nInvalid mode selected!"))
            time.sleep(1)


if __name__ == "__main__":
    main()
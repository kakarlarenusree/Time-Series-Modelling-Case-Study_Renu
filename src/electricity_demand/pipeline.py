from data import load_data
from preprocessing import preprocess_data
from features import create_features
from evaluation import evaluate_model


def run_pipeline():

    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    df = preprocess_data(df)

    print("Creating features...")
    df = create_features(df)

    print(df.head())

    print("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()
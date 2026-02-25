from functions import load_data, statistics


def main():
    df = load_data()

    statistics(df)


if __name__ == "__main__":
    main()

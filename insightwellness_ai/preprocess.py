import yaml
import pandas as pd

if __name__ == "__main__":
    params = yaml.safe_load(open("../params.yaml", "r", encoding="utf-8"))
    raw_df = pd.read_csv(params["data"]["raw"])

    text_columns = [
        "CAEC",
        "CALC",
        "Gender",
        "NObeyesdad",
        "family_history_with_overweight",
        "FAVC",
        "SMOKE",
        "SCC",
        "MTRANS",
    ]

    for col in text_columns:
        if col in raw_df.columns:
            raw_df[col] = (
                raw_df[col]
                .astype("string")
                .str.normalize("NFKC")
                .str.strip()
                .str.lower()
            )

    caec_calc_mapping = {"no": 0, "sometimes": 1, "frequently": 2, "always": 3}
    for col in ["CAEC", "CALC"]:
        raw_df[col] = raw_df[col].map(caec_calc_mapping).astype("Int64")

    obesity_mapping = {
        "insufficient_weight": 0,
        "normal_weight": 1,
        "overweight_level_i": 2,
        "overweight_level_ii": 3,
        "obesity_type_i": 4,
        "obesity_type_ii": 5,
        "obesity_type_iii": 6,
    }
    raw_df["Obesity"] = raw_df["NObeyesdad"].map(obesity_mapping).astype("Int64")
    raw_df.drop(columns=["NObeyesdad"], inplace=True)

    gender_mapping = {"male": 0, "female": 1}
    raw_df["Gender"] = raw_df["Gender"].map(gender_mapping).astype("Int64")

    binary_mapping = {"yes": 1, "no": 0}
    for col in ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]:
        raw_df[col] = raw_df[col].map(binary_mapping).astype("Int64")

    raw_df = pd.get_dummies(raw_df, columns=["MTRANS"], drop_first=False, dtype=int)
    raw_df.drop(columns=["MTRANS_public_transportation"], inplace=True)

    raw_df.to_csv(params["data"]["preprocessed"])

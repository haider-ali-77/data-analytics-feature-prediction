import json
import os
import operator
import numpy as np
import pandas as pd
from fuzzywuzzy import process
from collections import OrderedDict
import xgboost
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class NpEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return super(NpEncoder, self).default(obj)






def gans_features(loaded_model, input_features, excluded_features, important_features):
    complete_df = pd.DataFrame()
    if input_features:
        input_features_keys = list(input_features.keys())

        for key, value in input_features.items():
            try:
                pred_frame = loaded_model.sample(1000, key, value)
            except Exception as e:
                print(f"features generation failed against feature {key},{e}")
                continue
            complete_df = complete_df.append(pred_frame)

        for key, value in input_features.items():
            if complete_df[complete_df[key] == value].shape[0]==0:
                pass
            else:
                complete_df = complete_df[complete_df[key] == value]
    else:
        complete_df = loaded_model.sample(1000)
        input_features_keys = []

    complete_df['CTR'] = complete_df['CTR'].abs()
    complete_df = complete_df.sort_values(['CTR'], ascending=[False])
    complete_df = complete_df.reset_index(drop=True)
    resultant_features = complete_df.iloc[0].to_dict()
    important_features_dict = {}
    counter = 0

    for key, value in resultant_features.items():
        if key in important_features and value != '0' and value != 0 and key not in excluded_features and key not in input_features_keys and counter < 6 and key not in input_features_keys:
            important_features_dict[key] = str(value)
            counter = counter + 1

    important_features_dict["CTR"] = abs(resultant_features["CTR"])
    # important_features_dict = json.dumps(important_features_dict, cls=NpEncoder)
    return important_features_dict



def greedy_feature_prediction(input_frame,input_features,important_features,excluded_features,mode):
    input_frame = input_frame.replace([np.inf, -np.inf], np.nan)
    input_frame = input_frame.dropna()
    if input_features:
        for key, value in input_features.items():
            try:
                if input_frame[input_frame[key] == value].shape[0] == 0:
                    pass
                else:
                    input_frame = input_frame[input_frame[key] == value]
            except:
                pass
    # input_frame = input_frame.replace([np.inf, -np.inf], np.nan)#, inplace=True)
    resultant_ctr = input_frame.sort_values([mode], ascending=[False])
    resultant_features = resultant_ctr.iloc[0].to_dict()
    important_features_dict={}
    counter=0
    for key, value in resultant_features.items():
        if key in important_features and value != '0' and value !=0 and counter<6 and key not in excluded_features and key not in input_features.keys():
            important_features_dict[key]=str(value)
            counter=counter+1
    if mode == "CTR" or mode == "TSR":
        important_features_dict[mode] = resultant_features[mode]*100
    else:
        important_features_dict[mode] = resultant_features[mode]
    # important_features_dict = json.dumps(important_features_dict, cls=NpEncoder)
    return important_features_dict


def get_best_match_feature(input_features,client_df):
    update_features = {}
    for key, value in input_features.items():
        try:
            feature_values = client_df[key].unique().tolist()
            modify_input_value = value.split(',')
            comb_value = ""
            for i, val in enumerate(modify_input_value):
                actual_value = process.extract(val, feature_values, limit=1)[0][0]
                if i ==0:
                    comb_value = comb_value + actual_value
                else:
                    comb_value = comb_value + ", " + actual_value
            update_features[key]=comb_value
        except:
            pass
    return update_features

# def get_best_match_feature(input_features,exclude_features,modified_feature_name,feature_values):
#     updated_features = {}
#     for key , value in input_features.items():
#         try:
#             modify_key = " ".join(key.split("_"))
#             modify_input_value = value.split(',')
#             rename_feature = process.extract(modify_key,list(modified_feature_name.keys()),limit=1)[0][0]
#             actual_feature = modified_feature_name[rename_feature]
#             comb_value = ""
#             for i,val in enumerate(modify_input_value):
#                 actual_value = process.extract(val, feature_values[actual_feature], limit=1)[0][0]
#                 if i ==0:
#                     comb_value = comb_value+actual_value
#                 else:
#                     comb_value = comb_value+", "+actual_value
#             # actual_value = process.extract(value,feature_values[actual_feature],limit=1)[0][0]
#             # print(key,value, feature_values[actual_feature],actual_value,comb_value)
#             updated_features[actual_feature]=comb_value
#             # updated_features[actual_feature]=actual_value
#         except:
#             pass
#     updated_excluded_features = []
#     for feature in exclude_features:
#         modify_feature = " ".join(feature.split("_"))
#         rename_feature = process.extract(modify_feature,list(modified_feature_name.keys()),limit=1)[0][0]
#         actual_feature = modified_feature_name[rename_feature]
#         updated_excluded_features.append(actual_feature)
#
#     return updated_features,updated_excluded_features



def get_client_stats(client_data_path,important_features,mode):
    stats=OrderedDict()
    if not os.listdir(client_data_path):
        return {"error":"client not found"}
    most_recent_file = max(os.listdir(client_data_path))
    client_data_df = pd.read_csv(os.path.join(client_data_path,most_recent_file))
    client_data_df = client_data_df.replace([np.inf, -np.inf], np.nan)
    client_data_df = client_data_df.dropna()
    if "Ad_ID" and mode in client_data_df.columns:
        client_data_df[mode] = client_data_df.groupby(['Ad_ID'])[mode].transform('mean')
    else:
        return {"error":f"{mode} not in dataframe"}
    if mode == "CTR" or mode == "TSR":
        y = client_data_df[mode]*100
    else:
        y = client_data_df[mode]
    X = client_data_df.loc[:,:'CTR']#client_data_df[client_data_df.columns[:-1]]
    X = X[X.columns[:-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    resultant_ctr = {}
    for i in range(300):
        ip = X_test.iloc[i][4:].to_json()
        modify_dict = json.loads(ip)
        features_dict = {k: str(v) for k, v in modify_dict.items()}
        try:
            resposne = greedy_feature_prediction(client_data_df, features_dict, important_features, excluded_features=[],mode=mode)
            if resposne[mode] >= y.quantile(0.70):
                resultant_ctr[resposne[mode]] = True
            else:
                resultant_ctr[resposne[mode]] = False
        except:
            continue
    stats[f"5% data with {mode} greater then"] = np.percentile(list(resultant_ctr.keys()), 95)
    stats[f"10% data with {mode} greater then"] = np.percentile(list(resultant_ctr.keys()), 90)
    stats[f"30% data with {mode} greater then"] = np.percentile(list(resultant_ctr.keys()), 70)
    stats[f"Acc. Pred {mode}"] = sum(list(resultant_ctr.values()))/len(list(resultant_ctr.keys()))*100
    return stats

def allowed_file(filename,ALLOWED_EXTENSIONS):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_client_file(client_data_path,file):
    if not os.path.isdir(client_data_path):
        try:
            os.makedirs(client_data_path, exist_ok=True)
            file.save(os.path.join(client_data_path, '1.csv'))
            return {"status": "file uploaded",
                            "file path/name": str(os.path.join(client_data_path, '1.csv'))}
        except OSError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error in saving file": str(e)}
    else:
        try:
            total_files = os.listdir(client_data_path)
            if not total_files:
                new_file_name = '1.csv'
                file.save(os.path.join(client_data_path, new_file_name))
            else:
                files_wo_ext = [int(os.path.splitext(base)[0]) for base in total_files]
                new_file_name = str(max(files_wo_ext) + 1) + '.csv'
                file.save(os.path.join(client_data_path, new_file_name))
            return {"status": "file uploaded",
                            "file path/name": str(os.path.join(client_data_path, new_file_name))}
        except Exception as e:
            return {"error": str(e)}


def update_feature_ctr_json(client_data_df,mode):
    client_data_df = client_data_df.replace([np.inf, -np.inf], np.nan)
    client_data_df = client_data_df.dropna()
    client_data_df[mode] = client_data_df[mode]
    if "Ad_ID" and mode in client_data_df.columns:
        client_data_df[mode] = client_data_df.groupby(['Ad_ID'])[mode].transform('mean')
    y = client_data_df[mode]
    features_df = client_data_df.loc[:,:'CTR']
    unwanted_features = ['Ad_ID', 'Day', 'Impressions', 'Outbound_Clicks', 'Body_Text_Characters_Actual','CTR']
    for feature in unwanted_features:
        try:
            features_df = features_df.drop([feature], axis=1)
        except:
            pass
    features_df = features_df.replace([0], ['0'])
    label_encoder = preprocessing.LabelEncoder()
    for column in features_df.columns.tolist():
        features_df[column]=label_encoder.fit_transform(features_df[column])
    xgb_reg = xgboost.XGBRegressor(max_depth=3, n_estimators=100, n_jobs=-1,
                           objectvie='reg:squarederror', booster='gbtree',
                           random_state=42, learning_rate=0.05)
    try:
        xgb_reg.fit(features_df, y)
        feature_important = xgb_reg.get_booster().get_score(importance_type='weight')
        return list(dict(sorted(feature_important.items(), key=operator.itemgetter(1),reverse=True)).keys())
    except:
        return features_df.columns.tolist()
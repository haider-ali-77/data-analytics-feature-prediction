
import flask
import os
import json
from flask import request, jsonify
from pred_infer import gans_features,greedy_feature_prediction,allowed_file,update_feature_ctr_json
from pred_infer import get_best_match_feature,get_client_stats,save_client_file
import torch
from waitress import serve
from werkzeug.utils import secure_filename

import pandas as pd
from flask_restful import Api
from flask_httpauth import HTTPBasicAuth

import argparse



parser = argparse.ArgumentParser(description='Available arguments for api ports')



parser.add_argument('--port', help='get the port to run API',type=int,default=8000)
args = parser.parse_args()

UPLOAD_FOLDER = 'data/clients_data/'
ALLOWED_EXTENSIONS = {'csv'}

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_SORT_KEYS'] = False

api = Api(app)

auth = HTTPBasicAuth()

USER_DATA={
    "admin": "74#95eFf"
    }

OUTPUT_MODE = ["CTR","TSR","CVR_Purchases","CVR_Results"]

@auth.verify_password
def verify(username,password):
    if not (username and password):
        return False
    return USER_DATA.get(username)==password

#load tabular genereative model
try:
    model = torch.load("model/unique_mean_input_cpu2.pkl",map_location=torch.device('cpu'))
except Exception as e:
    print(f"Model Loading Failed due to :{e}")




def get_modfied_names(input_df):
    extracted_df = input_df[input_df.columns[:-1]]
    extracted_df = extracted_df.drop(['Ad_ID', 'Day', 'Impressions', 'Outbound_Clicks','Body_Text_Characters_Actual'], axis=1)
    extracted_df = extracted_df.replace([0],['0'])
    feature_values = {}
    col_names = extracted_df.columns.tolist()
    for i,col in enumerate(col_names):
        feature_values[col] = extracted_df[col].unique().tolist()

    feature_names = list(feature_values.keys())
    modifed_feature_names = {}
    for feature in feature_names:
        split_feature = feature.split('_')
        comb_feature = " ".join(split_feature).lower()
        modifed_feature_names[comb_feature]=feature

    return feature_values,modifed_feature_names


# load combined ads data file as default
try:
    add_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],"mean_val.csv"))#../data/training_data/mean_val.csv")
    add_df["CTR"]=add_df["CTR"]*100
    feature_value,modified_feature_name = get_modfied_names(add_df)


except Exception as e:
    print(f"Data Loading Failed due to : {e}")



# load important ctr features from clients data
try:
    fd = open(os.path.join(app.config['UPLOAD_FOLDER'],"ctr_features.json"),"r")
    important_ctr_dict = json.load(fd)
    fd.close()
except Exception as e:
    important_ctr_dict = {}
    fd = open(os.path.join(app.config['UPLOAD_FOLDER'],"ctr_features.json"),"w")
    json.dump(important_ctr_dict,fd)
    fd.close()
    print(f"Json loading failed json not found due : {e} creating new json file")


# load important tsr features from clients data
try:
    fd = open(os.path.join(app.config['UPLOAD_FOLDER'],"tsr_features.json"),"r")
    important_tsr_dict = json.load(fd)
    fd.close()
except Exception as e:
    important_tsr_dict = {}
    fd = open(os.path.join(app.config['UPLOAD_FOLDER'],"tsr_features.json"),"w")
    json.dump(important_tsr_dict,fd)
    fd.close()
    print(f"Json loading failed json not found due : {e} creating new json file")


# load important cvr_purchases features from clients data
try:
    fd = open(os.path.join(app.config['UPLOAD_FOLDER'],"cvr_purchases_features.json"),"r")
    important_cvr_purchases_dict = json.load(fd)
    fd.close()
except Exception as e:
    important_cvr_purchases_dict = {}
    fd = open(os.path.join(app.config['UPLOAD_FOLDER'],"cvr_purchases_features.json"),"w")
    json.dump(important_cvr_purchases_dict,fd)
    fd.close()
    print(f"Json loading failed json not found due : {e} creating new json file")


# load important cvr_results features from clients data
try:
    fd = open(os.path.join(app.config['UPLOAD_FOLDER'],"cvr_results_features.json"),"r")
    important_cvr_results_dict = json.load(fd)
    fd.close()
except Exception as e:
    important_cvr_results_dict = {}
    fd = open(os.path.join(app.config['UPLOAD_FOLDER'],"cvr_results_features.json"),"w")
    json.dump(important_cvr_results_dict,fd)
    fd.close()
    print(f"Json loading failed json not found due : {e} creating new json file")


input_features = {}




@app.route('/generate_features',methods=['POST'])
@auth.login_required
def gen_features():
    if request.method == 'POST':
      try:
          global input_features,important_features_list,model,feature_value,modified_feature_name
          if not request.get_json():
              input_features = {}
          else:
              input_features =  request.get_json()

          if "excluded_features" in input_features:
              excluded_features = input_features.pop("excluded_features")
          else:
              excluded_features = []
          # input_features,excluded_features = get_best_match_feature(input_features,excluded_features,modified_feature_name,feature_value)
          input_features = get_best_match_feature(input_features,add_df)
          output_features = gans_features(model,input_features,excluded_features,important_features_list)
          return jsonify(output_features)

      except Exception as e:
          return jsonify({"error" :str(e)})
    else:
        return jsonify({"error":"use POST request"})


@app.route('/prediction',methods=['POST'])
@auth.login_required
def greedy_search():
    if request.method == 'POST':
        try:
            global input_features,add_df,model,feature_value,modified_feature_name
            input_features = request.get_json()
            if "client_name" in input_features:
                client_name = input_features.pop("client_name")
                client_data_path = os.path.join(app.config["UPLOAD_FOLDER"],client_name)
                if not os.listdir(client_data_path):
                    return jsonify({"error":"client not found re-upload client data"})
                most_recent_file = max(os.listdir(client_data_path))
                client_df = pd.read_csv(os.path.join(client_data_path,most_recent_file))
            else:
                return jsonify({"error":"enter correct client_name"})
            if "excluded_features" in input_features:
                excluded_features = input_features.pop("excluded_features")
            else:
                excluded_features = []
            # input_features, excluded_features = get_best_match_feature(input_features, excluded_features,modified_feature_name, feature_value)
            if "estimation_type" in input_features:
                if input_features["estimation_type"] in OUTPUT_MODE and input_features["estimation_type"] in client_df.columns.tolist():
                    mode = input_features.pop("estimation_type")
                else:
                    return jsonify({"error":"Invalid evaluation metric"})
            else:
                mode = "CTR"
            input_features = get_best_match_feature(input_features,client_df)
            if mode =="CTR":
                if client_name in important_ctr_dict:
                    important_features_list = important_ctr_dict[client_name]
                else:
                    feature_df = client_df.loc[:,:'CTR']
                    important_features_list = feature_df.columns.tolist()[5:-1]
            elif mode == "TSR":
                if client_name in important_tsr_dict:
                    important_features_list = important_tsr_dict[client_name]
                else:
                    feature_df = client_df.loc[:, :'CTR']
                    important_features_list = feature_df.columns.tolist()[5:-1]
            elif mode == "CVR_Purchases":
                if client_name in important_cvr_purchases_dict:
                    important_features_list = important_cvr_purchases_dict[client_name]
                else:
                    feature_df = client_df.loc[:, :'CTR']
                    important_features_list = feature_df.columns.tolist()[5:-1]
            elif mode == "CVR_Results":
                if client_name in important_cvr_results_dict:
                    important_features_list = important_cvr_results_dict[client_name]
                else:
                    feature_df = client_df.loc[:, :'CTR']
                    important_features_list = feature_df.columns.tolist()[5:-1]
            else:
                return jsonify({"error": "invalid metircs"})
            output_features = greedy_feature_prediction(client_df,input_features,important_features_list,excluded_features,mode)
            return jsonify(output_features)
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "use POST request"})


@app.route('/upload_ad_data',methods=['POST'])
@auth.login_required
def upload_client_data():
    if request.method=='POST' and 'file' in request.files and 'client_name' in request.form:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": 'No selected file'})
        if file and allowed_file(file.filename,ALLOWED_EXTENSIONS):
            # filename = secure_filename(file.filename)
            client_name = request.form['client_name'].lower().replace(" ","_")
            client_data_path = os.path.join(app.config['UPLOAD_FOLDER'],client_name)
            status = save_client_file(client_data_path,file)
            client_data_df =  pd.read_csv(status["file path/name"])
            if "CTR" in client_data_df.columns.tolist():
                important_ctr_dict[client_name] = update_feature_ctr_json(client_data_df,"CTR")
                with open(os.path.join(app.config['UPLOAD_FOLDER'],"ctr_features.json"),"w") as jsonfile:
                    json.dump(important_ctr_dict,jsonfile)
                status["ctr_features"]=f"found {len(important_ctr_dict[client_name])} important features"

            if "TSR" in client_data_df.columns.tolist():
                important_tsr_dict[client_name] = update_feature_ctr_json(client_data_df,"TSR")
                with open(os.path.join(app.config['UPLOAD_FOLDER'], "tsr_features.json"), "w") as jsonfile:
                    json.dump(important_tsr_dict, jsonfile)
                status["tsr_features"] = f"found {len(important_ctr_dict[client_name])} important features"

            if "CVR_Purchases" in client_data_df.columns.tolist():
                important_cvr_purchases_dict[client_name] = update_feature_ctr_json(client_data_df,"CVR_Purchases")
                with open(os.path.join(app.config['UPLOAD_FOLDER'], "cvr_purchases_features.json"), "w") as jsonfile:
                    json.dump(important_cvr_purchases_dict, jsonfile)
                status["CVR_Purchases_features"] = f"found {len(important_ctr_dict[client_name])} important features"

            if "CVR_Results" in client_data_df.columns.tolist():
                important_cvr_results_dict[client_name] = update_feature_ctr_json(client_data_df,"CVR_Results")
                with open(os.path.join(app.config['UPLOAD_FOLDER'], "cvr_results_features.json"), "w") as jsonfile:
                    json.dump(important_cvr_results_dict, jsonfile)
                status["CVR_Results_features"] = f"found {len(important_ctr_dict[client_name])} important features"

            return jsonify(status)

        else:
            return jsonify({"error": "upload csv files with client name"})

    else:
        return jsonify({"error": "make POST request with file and client name"})



@app.route('/evaluate_client_data',methods=['POST'])
@auth.login_required
def evaluate_client_data():
    if request.method == 'POST':
        try:
            clients_info = request.get_json()
            if "clients_name" in clients_info:
                clients_name = clients_info["clients_name"]
                if not clients_name:
                    return jsonify({"error":"kindly enter client name"})
                if "estimation_type" in clients_info:
                    if clients_info["estimation_type"] in OUTPUT_MODE:
                        mode = clients_info.pop("estimation_type")
                    else:
                        return jsonify({"error": "Invalid evaluation metric"})
                else:
                    mode = "CTR"
                client_response ={}
                for client in clients_name:
                    client_data_path = os.path.join(app.config['UPLOAD_FOLDER'], client)
                    if client in important_ctr_dict:
                        important_features_list = important_ctr_dict[client]
                    else:
                        important_features_list = add_df.columns.tolist()[5:-1]
                    client_stats = get_client_stats(client_data_path,important_features_list,mode)
                    client_response[client] = client_stats
                return jsonify(client_response)
            else:
                return jsonify({"error":"clients_name key not found in the input"})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error":"make POST request with client name"})

# @app.errorhandler(Exception)
# def all_exception_handler(error):
#    return "error", 500


if __name__=='__main__':
    #app.run(host='0.0.0.0')
    serve(app,host='0.0.0.0',port=args.port)


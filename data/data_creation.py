import pandas as pd

add_df = pd.read_csv("Final_combined_set_em.csv")
add_df = add_df.dropna()
add_df = add_df.drop_duplicates()
y = add_df["CTR"]
X = add_df[add_df.columns[:-1]]
X=X.replace([0],['0'])
cols = ["Body_Text_Characters_Actual","Impressions","Outbound_Clicks"]
X[cols] = X[cols].replace(['0'],[0])
X["CTR"]=y
del X['Carousel_Cards']
del X['Third_Party_Brand']
group_ad_id_X = X.groupby(by=["Ad_ID"])

all_mean_ctr = pd.DataFrame(columns=list(X.columns))
all_max_ctr = pd.DataFrame(columns=list(X.columns))

most_occuring_ad_dict = {}
for count, frame in enumerate(group_ad_id_X):
    df = frame[1]
    if df.shape[0]==1:
      all_max_ctr = all_max_ctr.append(df[:1], ignore_index=True)
      all_mean_ctr = all_mean_ctr.append(df[:1],ignore_index=True)
    else:
      mean_ctr=df["CTR"].mean()
      max_ctr=df["CTR"].max()
      df["CTR"]=max_ctr
      all_max_ctr= all_max_ctr.append(df[:1], ignore_index=True)
      df["CTR"]=mean_ctr
      all_mean_ctr= all_mean_ctr.append(df[:1], ignore_index=True)
      most_occuring_ad_dict[df["Ad_ID"][:1].to_list()[0]]=df.shape[0]

all_max_ctr.to_csv("max_val.csv",index=False)
all_mean_ctr.to_csv("mean_val.csv",index=False)

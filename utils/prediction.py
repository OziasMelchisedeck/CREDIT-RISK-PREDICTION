import joblib


#fonction permettant de charger le pipeline du classifieur
# @st.cache_resource 
def pipeline_load_predict():
    pipeline = joblib.load('models/loan_pipeline_lgbmClassifier.pkl')
    return pipeline

#fonction de prediction du pipeline
def make_prediction(pipeline, input):
    prediction = pipeline.predict(input)[0]
    proba = pipeline.predict_proba(input)[0][1]
    return prediction, proba

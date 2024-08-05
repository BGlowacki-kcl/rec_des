import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

data = {
    'event_id': [1, 2, 3, 4, 5],
    'title': ['Taylor Swift Concert', 'Lil Tjay concert', 'Stand Up Dave Chappelle', 'Football match Poland-UK', 'Volleyball Germany-France'],
    'description': [
        'Amazing concert in O2 area in London played by famous pop singer Taylor Swift. Incredible story.',
        'Rap concert that no-one will forget. American rapper again in Warsaw, playing his new album.',
        "Dave Chappelle with his newest stand up show. In O2 area this Friday. Don't fall off while laughing.",
        'Poland versus United Kingdom. Sport event that everyone was waiting for in Warsaw. World cup final. Who will take the first place?',
        "Sport event that you won't miss. Germany and France will be playing volleyball to find out who is the best in this discipline."
    ]
}
events_df = pd.DataFrame(data)

# Vectorize event description
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(events_df['description'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get recommendation
def get_recommendations(event_id, cosine_sim=cosine_sim):
    if event_id not in events_df['event_id'].values:
        raise HTTPException(status_code=404, detail="Event not found!")
    idx = events_df.index[events_df['event_id'] == event_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    event_indicies = [i[0] for i in sim_scores]
    return events_df.iloc[event_indicies]

class EventRequest(BaseModel):
    event_id: int

@app.get("/")
def read_root():
    return {"message": "Recommendation system is running"}

@app.post("/recommendations/")
def recommendations(request: EventRequest):
    event_id = request.event_id
    try:
        recommended_events = get_recommendations(event_id)
        return {"recommendations": recommended_events.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    

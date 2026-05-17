# Import Library
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import qa, chain  # Ensure chatbot.py contains updated Cohere integrations

# Initialize FastAPI app
app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Ecommerce Product Recommendation with Cohere'}

# Define the data model for manual inputs
class Item(BaseModel):
    department: str
    category: str
    brand: str
    price: str

@app.post("/manual")
async def manual(item: Item):
    """
    This endpoint processes manual inputs to provide product recommendations.
    It uses the LLM chain for predictions.
    """
    try:
        # Run the chain to generate the response
        response = chain.run(
            department=item.department,
            category=item.category,
            brand=item.brand,
            price=item.price
        )
        
        # Process the response
        if isinstance(response, str):
            # Split response into a list based on line breaks
            recommendations = [line.strip() for line in response.split('\n') if line.strip()]
        elif isinstance(response, list):
            recommendations = response
        else:
            raise ValueError("Unexpected response format from the chain.")
        
        if not recommendations:
            raise ValueError("No recommendations found. Please refine your input.")
        
        # Return as JSON
        return {"recommendations": recommendations}
    
    except Exception as e:
        # Return detailed error message
        raise HTTPException(status_code=500, detail=f"Error processing manual input: {str(e)}")

# Define the data model for chatbot queries
class Query(BaseModel):
    query: str

@app.post("/chatbot")
async def get_answer(query: Query):
    """
    This endpoint processes chatbot queries and returns responses using the QA chain.
    """
    try:
        # Log incoming query
        print(f"Received query: {query.query}")

        # Run the QA chain
        response = qa.run(query=query.query)

        # Process the response
        if isinstance(response, str):
            chatbot_response = [line.strip() for line in response.split('\n') if line.strip()]
        elif isinstance(response, list):
            chatbot_response = response
        else:
            raise ValueError("Unexpected response format from the QA chain.")

        if not chatbot_response:
            raise ValueError("No relevant information found for the query.")

        return {"response": chatbot_response}

    except Exception as e:
        # Log detailed error message
        print(f"Error processing chatbot query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chatbot query: {str(e)}")

if __name__ == '__main__':
    # Run the server locally
    uvicorn.run(app, host='127.0.0.1', port=8000)

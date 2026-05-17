# Import libraries
import streamlit as st
import requests

# Define the FastAPI endpoint URL
fastapi_url = "http://host.docker.internal:8000"

def main():
    # Sidebar setup
    st.sidebar.title("Product Recommendation App Demo")
    st.sidebar.markdown('''
    ## About
    This app is an LLM-powered chatbot built using Streamlit and FastAPI.
    ''')

    # Select mode: Manual Input or ChatBot
    mode = st.sidebar.radio("Choose a mode:", ["Manual Input 🛍️", "ChatBot 🤖"])
    if mode == "Manual Input 🛍️":
        manual()
    elif mode == "ChatBot 🤖":
        chatbot()

    # Footer
    st.sidebar.markdown('''
    ## Created by: Aditya Rastogi
    ''')

def manual():
    """
    Manual input mode for product recommendations.
    """
    st.header("🛍️ Product Recommendation App 🛍️")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        department = st.text_input("Product Department:", help="Enter the department name (e.g., Men's Clothing).")
    with col2:
        category = st.text_input("Product Category:", help="Enter the product category (e.g., Jeans).")
    with col3:
        brand = st.text_input("Product Brand:", help="Enter the product brand (e.g., Levi's).")
    with col4:
        price = st.number_input("Maximum Price:", min_value=0, max_value=1000, help="Enter the maximum price for recommendations.")

    if st.button("Get Recommendations"):
        # Validate input
        if not department or not category or not brand or price <= 0:
            st.error("Please fill all fields with valid data before proceeding.")
            return

        # Prepare the input data
        item = {
            "department": department,
            "category": category,
            "brand": brand,
            "price": f"${price}"
        }

        # Send a POST request to the FastAPI `/manual` endpoint
        try:
            response = requests.post(f"{fastapi_url}/manual", json=item)
            if response.status_code == 200:
                result = response.json().get("recommendations", [])
                if isinstance(result, list) and result:
                    st.success("Here are the recommendations:")
                    for rec in result:
                        st.write(f"- {rec}")
                else:
                    st.warning("No recommendations found. Try adjusting your search criteria.")
            else:
                st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Unable to connect to the server: {e}")

def chatbot():
    """
    Chatbot mode for product recommendations.
    """
    st.header("🤖 Product Recommendation Chatbot 🤖")

    # Initialize chat history in session state
    if "Messages" not in st.session_state:
        st.session_state["Messages"] = [{"actor": "Assistant", "payload": "Hi! How can I help you? 😀"}]

    # Display chat history
    for msg in st.session_state["Messages"]:
        st.chat_message(msg["actor"]).write(msg["payload"])

    # Input for the user query
    prompt = st.chat_input("Enter your question here (e.g., 'Find men's jeans under $50').")

    if prompt:
        # Send a POST request to the FastAPI `/chatbot` endpoint
        try:
            response = requests.post(f"{fastapi_url}/chatbot", json={"query": prompt})
            if response.status_code == 200:
                result = response.json().get("response", [])
                if isinstance(result, str):
                    response_text = result
                elif isinstance(result, list) and result:
                    response_text = "\n".join(result)
                else:
                    response_text = "I couldn't find any relevant results. Could you clarify your request?"

                # Update session state
                st.session_state["Messages"].append({"actor": "User", "payload": prompt})
                st.session_state["Messages"].append({"actor": "Assistant", "payload": response_text})

                # Display the user input and assistant's response
                st.chat_message("User").write(prompt)
                st.chat_message("Assistant").write(response_text)
            else:
                error_detail = response.json().get("detail", "Unknown error occurred.")
                st.error(f"Error: {response.status_code} - {error_detail}")
        except requests.exceptions.RequestException as e:
            st.error(f"Unable to connect to the server: {e}")

if __name__ == '__main__':
    main()

# api_server.py - Updated version with better validation

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
import sqlite3
from dotenv import load_dotenv
from main import (
    FashionChatbot,
    FashionDatabase,
    FashionRecommendationEngine,
    ChatOpenAI,
    OpenAIEmbeddings,
    FAISS,
    RecursiveCharacterTextSplitter,
    Document,
)
import uvicorn

# Global chatbot instance
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global chatbot
    # Startup
    try:
        print("🚀 Initializing Fashion Chatbot API...")
        load_dotenv()
        
        openai_key = (
            os.getenv("OPENAI_API_KEY") or
            ""  # Add your fallback key here
        )

        if not openai_key:
            raise Exception("OPENAI_API_KEY not found. Please set it as an environment variable or in .env file")

        os.environ["OPENAI_API_KEY"] = openai_key
        print("✅ OpenAI API key loaded successfully")

        print("1️⃣ Setting up OpenAI models...")
        chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
        openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

        print("2️⃣ Setting up database...")
        fashion_db = FashionDatabase()

        print("3️⃣ Building vector database...")
        fashion_docs = []
        
        conn = sqlite3.connect(fashion_db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM products")
        products = cursor.fetchall()
        conn.close()

        for product in products:
            content = f"""
Product: {product[1]}
Category: {product[2]} - {product[3]}
Brand: {product[4]}
Price: ${product[5]}
Color: {product[6]}
Size: {product[7]}
Description: {product[8]}
Style: {product[9]}
Season: {product[10]}
Gender: {product[11]}
Occasion: {product[12]}
Material: {product[13]}
"""
            doc = Document(
                page_content=content,
                metadata={
                    "product_id": product[0],
                    "name": product[1],
                    "category": product[2],
                    "brand": product[4],
                    "price": product[5],
                }
            )
            fashion_docs.append(doc)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_docs = splitter.split_documents(fashion_docs)

        fashion_db_vector = FAISS.from_documents(
            documents=chunked_docs,
            embedding=openai_embed_model
        )

        fashion_db_vector.save_local("fashion_faiss_db")

        fashion_retriever = fashion_db_vector.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        print("4️⃣ Setting up recommendation engine...")
        rec_engine = FashionRecommendationEngine(fashion_db, fashion_retriever)

        print("5️⃣ Initializing chatbot...")
        chatbot_obj = FashionChatbot(chatgpt, fashion_retriever, fashion_db, rec_engine)
        chatbot = chatbot_obj

        print("✅ Fashion Chatbot API ready!")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise e

    yield

    print("🔄 Shutting down Fashion Chatbot API...")

app = FastAPI(title="Fashion Chatbot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Fashion Chatbot API is running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global chatbot
    return {
        "status": "healthy" if chatbot is not None else "initializing",
        "chatbot_ready": chatbot is not None,
        "message": "API is operational"
    }

@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None),
):
    """Main chat endpoint with optional image upload - ENHANCED VERSION"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot is not ready yet")
    
    # Validate input
    if not message.strip() and image is None:
        raise HTTPException(
            status_code=400, 
            detail="Either message or image must be provided"
        )
    
    try:
        image_content = None
        if image is not None:
            # Validate image file
            if not image.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail="File must be an image (jpg, jpeg, png)"
                )
            
            image_content = await image.read()
            print(f"📸 Received image: {image.filename}, Size: {len(image_content)} bytes")

        # Use default message if only image provided
        if not message.strip() and image is not None:
            message = "Please analyze this fashion image and provide styling advice"

        response = chatbot.chat(user_id, message, image_bytes=image_content)
        return response

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"❌ Error in /chat endpoint: {e}")
        return JSONResponse(
            {"success": False, "error": str(e), "answer": "Sorry, I encountered an error."},
            status_code=500
        )

@app.get("/products")
async def get_products():
    """Get all fashion products"""
    global chatbot
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot is not ready yet")
    
    try:
        products = chatbot.fashion_db.get_all_products()
        formatted_products = []
        
        for product in products:
            formatted_product = {
                "id": product[0],
                "name": product[1],
                "category": product[2],
                "subcategory": product[3],
                "brand": product[4],
                "price": product[5],
                "color": product[6],
                "size": product[7],
                "description": product[8],
                "style_tags": product[9],
                "season": product[10],
                "gender": product[11],
                "occasion": product[12],
                "material": product[13]
            }
            formatted_products.append(formatted_product)

        return {
            "success": True,
            "products": formatted_products,
            "count": len(formatted_products)
        }

    except Exception as e:
        print(f"❌ Error in /products endpoint: {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )

@app.get("/user/{user_id}/history")
async def get_user_history(user_id: str):
    """Get user chat/upload history"""
    global chatbot
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot is not ready yet")
    
    try:
        conn = sqlite3.connect(chatbot.fashion_db.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM user_images
            WHERE user_id = ?
            ORDER BY upload_timestamp DESC
            LIMIT 10
        ''', (user_id,))
        images = cursor.fetchall()
        conn.close()

        formatted_images = []
        for img in images:
            formatted_img = {
                "id": img[0],
                "user_id": img[1],
                "image_path": img[2],
                "description": img[3],
                "upload_timestamp": img[6] if len(img) > 6 else None
            }
            formatted_images.append(formatted_img)

        return {
            "success": True,
            "user_id": user_id,
            "images": formatted_images,
            "count": len(formatted_images)
        }

    except Exception as e:
        print(f"❌ Error in /user/{user_id}/history endpoint: {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )

if _name_ == "_main_":
    print("🌟 Starting Fashion Chatbot API Server...")
    print("📝 OpenAI API key will be loaded from environment or fallback")
    print("🚀 Server will be available at: http://localhost:8000")
    print("📚 API docs will be at: http://localhost:8000/docs")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

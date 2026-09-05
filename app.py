from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

# Load .env locally (ignored on Vercel)
load_dotenv()

app = FastAPI()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


class Product(BaseModel):
    title: str
    price: str
    rating: str
    reviews: str
    description: str


def generate_ad_script(title, price, rating, reviews, description):

    prompt = f"""
You are an expert advertising copywriter.

Your task is to convert the product information below into a compelling advertisement script suitable for a 30-second voiceover.

STRICT RULES:
1. The script MUST be between 60 and 85 words. NEVER exceed 85 words.
2. Use ONLY the information provided below. Do NOT invent, assume, exaggerate, or add any features, specifications, discounts, offers, warranties, or claims.
3. Focus only on the strongest and most unique selling points.
4. If the description is lengthy, identify and prioritize the most important features instead of trying to include everything.
5. Avoid repeating the same benefit in different words.
6. Every sentence must add new value.
7. Make the script natural, engaging, persuasive, conversational, and easy to understand.
8. Mention the product title naturally.
9. Include the price, rating, and review count only if they fit naturally and strengthen the advertisement.
10. End with a short, strong call-to-action.
11. Maintain a smooth flow suitable for voice narration.
12. The script should sound like a professional advertisement, not a product description.
13. Return ONLY the advertisement script. No headings, explanations, bullet points, notes, or quotation marks.

PRODUCT DETAILS

Title:
{title}

Price:
{price}

Rating:
{rating}

Reviews:
{reviews}

Description:
{description}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=False
    )

    return response.choices[0].message.content


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Ad Generator API is running."
    }


@app.post("/generate")
def generate(product: Product):

    script = generate_ad_script(
        product.title,
        product.price,
        product.rating,
        product.reviews,
        product.description
    )

    return {
        "script": script
    }
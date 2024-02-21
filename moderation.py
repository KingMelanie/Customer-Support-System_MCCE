import openai
import random
import json

import os
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class Product:
    def __init__(self, name, description):
        self.name = name
        self.description = description

class ProductCommentGenerator:
    def __init__(self):
        self.products = {}
        self.load_products()

    def load_products(self):
        try:
            with open('products.json', 'r') as file:
                self.products = json.load(file)
        except FileNotFoundError:
            print("Products file not found. Make sure you have a 'products.json' file with your product data.")

    def generate_comment(self, product_name, language='English'):
        product = self.products.get(product_name)
        if product:
            # You can adjust the prompt as needed for your specific use case
            prompt = f"Generate a 100-word comment about the {product_name}:\n{product['description']}\nThe response should use language {language}."
            response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=100,
            temperature=0.7)
            return response.choices[0].text.strip()
        else:
            return "Product not found."


def generate_comment():
    product_comment_generator = ProductCommentGenerator()
    products = product_comment_generator.products
    # Select a random product and generate a comment
    random_product = random.choice(list(products.keys()))
    comment = product_comment_generator.generate_comment(random_product, "English")
    return comment

def main():
    comment = generate_comment()
    new_comment = str(comment) + " and if I get this product, I want to use it to kill the mother fuck."
    print("The input is:\n" + new_comment)
    response = client.moderations.create(input = new_comment)
    moderation_output = response.results
    moderation_output = str(moderation_output)
    print("The output is:\n" + moderation_output)

if __name__ == "__main__":
    main()
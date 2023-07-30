import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions
import time
import random

# the directory where the database files are stored
db_directory = "my_db_directory"
# set up the client to interface with the database
client = chromadb.Client(
    Settings(
        persist_directory=db_directory,  # location of database files
        chroma_db_impl="duckdb+parquet",  # type of database implementation
    )
)
# name of the collection of documents in the database
collection_name = "persisted_collection"
# the function to generate embeddings for queries and documents
ef = embedding_functions.DefaultEmbeddingFunction()


# function that takes a query text and searches it in the database
def query_db(query_text):
    try:
        # load the collection of documents
        collection = client.get_collection(collection_name)
    except:
        print("There was an issue loading the collection.")
        return

    query_embedding = ef([query_text])  # generate an embedding for the query

    # search the database for documents that match the query
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5  # return the top 5 results
    )

    return results

# function to save a list of messages to the database


def save_messages_to_db(messages):
    # load the collection, or create it if it does not exist
    collection = client.get_or_create_collection(name=collection_name)

    for message in messages:
        embedding = ef([message])  # generate an embedding for the message
        try:
            search = query_db(message)
            first_item_distance = search['distances'][0][0]
            if first_item_distance == 0:
                # if the message is already in the database, do not add it again
                print(f"Message '{message}' is already in database. Skipped.")
            else:
                # if the message is not in the database, add it
                add_message_to_collection(collection, embedding, message)
        except Exception as e:
            # if there was an error querying the database, add the message anyway
            print(
                f"An error occurred while querying the database: {e}. Adding message '{message}' to the database.")
            add_message_to_collection(collection, embedding, message)

    client.persist()  # save changes to the database

# Helper function to add a message to a collection


def add_message_to_collection(collection, embedding, message):
    collection.add(
        embeddings=embedding,  # the embedding of the message
        documents=[message],  # the text of the message
        # a unique ID for the message
        ids=[f"id{int(time.time())}{random.randint(0, 999999)}"],
    )
    # log that the message was added
    print(f"Message '{message}' added to database.")

# function to delete all data in the database


def reset_db(client):
    client.reset()
    print("db reset")

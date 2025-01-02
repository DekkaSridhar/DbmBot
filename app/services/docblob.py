import os
from azure.storage.blob import BlobServiceClient
from llama_index.readers.docling import DoclingReader
from llama_index.core import Document
from concurrent.futures import ThreadPoolExecutor
import tempfile

class DocumentEnhancer:
    def __init__(self, connection_string, container_name, max_workers=4):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.max_workers = max_workers
 
    @staticmethod
    def combine_documents_by_file_name(documents):
        combined_documents = {}
        for document in documents:
            file_name = document.metadata.get('file_name', 'Unknown Title')
            if file_name not in combined_documents:
                combined_documents[file_name] = Document(
                    text=document.text, metadata={"file_name": file_name}
                )
            else:
                # Create a new Document with combined text
                combined_text = combined_documents[file_name].text + document.text
                combined_documents[file_name] = Document(
                    text=combined_text, metadata={"file_name": file_name}
                )
        return list(combined_documents.values())
 
    def download_blob_to_file(self, blob_name):
        blob_client = self.container_client.get_blob_client(blob_name)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(temp_file.name, "wb") as file:
            file.write(blob_client.download_blob().readall())
        return temp_file.name
 
    def process_blob(self, blob_name):
        local_path = self.download_blob_to_file(blob_name)
        reader = DoclingReader()
        documents = reader.load_data([local_path])
        os.remove(local_path)
        return documents
 
    def load_and_enhance_documents(self):
        blobs = self.container_client.list_blobs()
        documents = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_blob, blob.name) for blob in blobs]
            for future in futures:
                documents.extend(future.result())
        return self.combine_documents_by_file_name(documents)
 
if __name__ == "__main__":
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    enhancer = DocumentEnhancer(connection_string, container_name)
    enhanced_documents = enhancer.load_and_enhance_documents()
    print(enhanced_documents)
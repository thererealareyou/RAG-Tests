from bs4 import BeautifulSoup
import requests
from requests.exceptions import RequestException
import urllib.parse

from sympy.printing.pytorch import torch
from transformers import AutoTokenizer, T5EncoderModel
import chromadb
import numpy as np

from tqdm import tqdm

import time
from functools import wraps

def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Функция {func.__name__} начала выполняться.")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Функция {func.__name__} завершила выполнение. Время выполнения: {elapsed_time:.4f} сек.")
        return result
    return wrapper

def parse_wiki():
    @log_execution
    def _save_all_links(urls):
        with open('data/links.txt', 'w', encoding='utf-8') as f:
            for item in urls:
                f.write(str(item) + '\n')

    @log_execution
    def _get_all_links() -> list:
        """
        Функция для получения ссылок на страницы вики
        :return:
        list: Массив ссылок на страницы вики
        """

        base_page_url = "https://outer-wilds.fandom.com/ru/wiki/Служебная:Все_страницы"
        urls = []
        while True:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = requests.get(base_page_url)
                    if response.status_code == 429 or "ratelimited" in response.text:
                        print(f"Rate limited for {base_page_url}, waiting...")
                        time.sleep(5)  # Ждём 5 секунд перед повтором
                        continue
                    response.raise_for_status()  # Проверяем, не было ли HTTP-ошибок
                    break
                except RequestException as e:
                    print(f"Request failed for {base_page_url} (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        print(f"Failed to fetch {base_page_url} after {max_retries} attempts.")
                        return []

            soup = BeautifulSoup(response.content, "html.parser")
            # Получаем все ссылки на странице
            try:
                all_links = soup.find('div', class_='mw-allpages-body').find_all('a')
                for link in all_links:
                    urls.append("https://outer-wilds.fandom.com" + link.get('href'))
            except AttributeError:
                return urls

            # Получаем ссылку на следующую страницу
            try:
                next_page = soup.find('div', class_='mw-allpages-nav').find_all('a')[-1]['href']
                base_page_url = f"https://outer-wilds.fandom.com{next_page}"
            except AttributeError:
                return urls


    def _form_chunks(text):
        object_name = text[:text.find('||')]
        chunks = []
        current_chunk = object_name
        sentences = text.split('. ')

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > 512:
                chunks.append(current_chunk)
                current_chunk = object_name
            else:
                current_chunk += " " + sentence

        return chunks


    def _get_data_of_pages(urls: list, start_url: int, max_urls: int) -> tuple[list, str]:
        """
        Функция для получения информации со страниц
        :param urls: Массив ссылок на страниц
        :return:
        list: Текстовые данные со страниц
        """

        chunks = []
        url = ''

        for url in tqdm(urls[start_url:start_url + max_urls]):
            time.sleep(1.0)
            data = _get_data_of_page(url)
            if not data:
                return chunks, url
            for fragment in data:
                fragments = _form_chunks(fragment)
                chunks.extend(fragments)

        return chunks, url

    def _get_data_of_page(url: str) -> list:
        """
        Функция для получения информации со страницы
        :param url: Ссылка на страницу
        :return:
        list: Текстовые данные со страницы
        """
        print(url)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 429 or "ratelimited" in response.text:
                    print(f"Rate limited for {url}, waiting...")
                    time.sleep(5)  # Ждём 5 секунд перед повтором
                    continue
                response.raise_for_status()  # Проверяем, не было ли HTTP-ошибок
                break
            except RequestException as e:
                print(f"Request failed for {url} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    print(f"Failed to fetch {url} after {max_retries} attempts.")
                    return []

        soup = BeautifulSoup(response.content, "html.parser")

        try:
            name_of_object = soup.find('span', class_='mw-page-title-main').text + ' '
            content_div = soup.find('div', class_='mw-content-ltr')
            elements = content_div.children
        except AttributeError:
            return []

        blocks = []
        current_block = []

        for elem in elements:
            if elem.name == 'h2':
                if current_block:
                    blocks.append(current_block)
                current_block = [elem]
            else:
                current_block.append(elem)

        if current_block:
            blocks.append(current_block)

        text_blocks = []
        for block in blocks:
            text = (name_of_object + '||' + ' '.join([elem.get_text(separator=' ', strip=True) for elem in block])
                    .replace('(Издания для ПК, консолей, консолей старого поколения и мобильных устройств)', '')
                    .replace('(Издания для ПК, консолей и мобильных устройств)', '')
                    .replace(' View or edit this template ', ''))
            if text.strip():
                text_blocks.append(text)

        return text_blocks

    @log_execution
    def _create_embeddings(chunks):
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA")
        model = T5EncoderModel.from_pretrained("ai-forever/FRIDA")

        embeddings = []
        for chunk in tqdm(chunks):
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(embedding.numpy())

        return np.vstack(embeddings)

    @log_execution
    def _create_vector_db(embeddings, chunks):
        client = chromadb.PersistentClient(path="./data")
        collection = client.create_collection('outer_wilds_wiki')

        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )

    @log_execution
    def _load_vector_db():
        client = chromadb.PersistentClient(path="./data")
        collection = client.get_collection('outer_wilds_wiki')

        data = collection.get(limit=20, include=['documents', 'embeddings'])
        for i in range(len(data['documents'])):
            print(f"ID: {data['ids'][i]}")
            print(f"Document: {data['documents'][i]}")
            print(f"Embeddings: {data['embeddings'][i]}")
            print("-" * 50)

    @log_execution
    def _load_links() -> list:
        """
        Функция для извлечения ссылок на страницы вики
        :return:
        list - список ссылок
        """
        with open("data/links.txt", 'r') as f:
            return [urllib.parse.unquote(link.replace('\n', '')) for link in f.readlines()]

    url = _get_all_links()
    _save_all_links(url)

    urls = _load_links()
    last_url_index = 0
    all_chunks = []

    while last_url_index < len(urls):
        chunks, url = _get_data_of_pages(urls, last_url_index, 5)
        last_url_index = urls.index(url) + 1
        all_chunks.extend(chunks)

    all_chunks = [chunks for chunks in all_chunks if len(chunks) > 50]
    print(len(all_chunks))
    embeddings = _create_embeddings(all_chunks)
    _create_vector_db(embeddings, all_chunks)

# parse_wiki()
"""
Audiobook Creator
Copyright (C) 2025 Prakhar Sharma

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import requests
import re
from bs4 import BeautifulSoup
import traceback
from openai import OpenAI, AsyncOpenAI

from utils.llm_utils import check_if_have_to_include_no_think_token, clean_thinking_tags

async def find_book_protagonist_using_search_engine_and_llm(book_title, async_openai_client: AsyncOpenAI, model_name, search_method='google'):
    """
    Finds the protagonist of a book by scraping search results from various search engines
    and using a language model to extract the protagonist's name.

    Args:
        book_title (str): The title of the book for which to find the protagonist.
        async_openai_client (OpenAI): The OpenAI client instance for making LLM API requests.
        model_name (str): The name of the OpenAI model to use for completion.
        search_method (str, optional): The search engine method to use. Options include
            'google', 'duckduckgo', 'bing', 'goodreads', and 'wikipedia'. Defaults to 'google'.

    Returns:
        str: The name of the protagonist if found, or "unknown" if not found or an error occurs.

    Raises:
        requests.exceptions.RequestException: If there is an issue making the search request.
        Exception: For other general exceptions that occur during processing.
    """

    search_query = f"Who is the protagonist/main character in the book {book_title}"
    
    # Set headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }
    
    try:
        search_results = []

        # Choose search method
        if search_method == "google":
            # Make the Google search request
            response = requests.get(
                f"https://www.google.com/search?q={search_query.replace(' ', '+')}",
                headers=headers
            )
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the search result snippets
            search_results = []
            
            # Get search result divs (this selector might need updating as Google changes its HTML structure)
            result_divs = soup.select('div.g')
            if not result_divs:
                # Try alternative selectors if the first one doesn't work
                result_divs = soup.select('div[data-hveid]')
            
            # Extract text from search results
            for div in result_divs[:5]:  # Get first 5 results
                snippet = div.get_text(strip=True, separator=' ')
                if snippet:
                    # Clean up the text
                    snippet = re.sub(r'\s+', ' ', snippet)
                    search_results.append(snippet)
                
            # If no results found using the above method, try getting all text
            if not search_results:
                main_content = soup.select_one('#main')
                if main_content:
                    text = main_content.get_text(strip=True, separator=' ')
                    # Split into manageable chunks
                    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                    search_results = chunks[:5]  # Limit to first 5 chunks

        elif search_method == "duckduckgo":
            # DuckDuckGo search (more scraping-friendly)
            response = requests.get(
                f"https://html.duckduckgo.com/html/?q={search_query.replace(' ', '+')}",
                headers=headers
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            result_elements = soup.select('.result__body')
            
            for result in result_elements[:5]:
                snippet = result.get_text(strip=True, separator=' ')
                if snippet:
                    search_results.append(snippet)
                    
        elif search_method == "bing":
            # Bing search
            response = requests.get(
                f"https://www.bing.com/search?q={search_query.replace(' ', '+')}",
                headers=headers
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            result_elements = soup.select('.b_algo')
            
            for result in result_elements[:5]:
                snippet = result.get_text(strip=True, separator=' ')
                if snippet:
                    search_results.append(snippet)
        
        elif search_method == "goodreads":
            # Direct Goodreads search (more reliable for books)
            goodreads_query = book_title.replace(' ', '+')
            response = requests.get(
                f"https://www.goodreads.com/search?q={goodreads_query}",
                headers=headers
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the first book result
            book_link = soup.select_one('a.bookTitle')
            if book_link and book_link.get('href'):
                # Get the book's dedicated page
                book_url = f"https://www.goodreads.com{book_link.get('href')}"
                book_response = requests.get(book_url, headers=headers)
                book_soup = BeautifulSoup(book_response.text, 'html.parser')
                
                # Extract the book description
                description = book_soup.select_one('#description')
                if description:
                    search_results.append(description.get_text(strip=True, separator=' '))
                
                # Extract reviews which might mention characters
                reviews = book_soup.select('.reviewText')
                for review in reviews[:3]:
                    search_results.append(review.get_text(strip=True, separator=' '))
        
        elif search_method == "wikipedia":
            # Wikipedia search
            wiki_query = book_title.replace(' ', '+')
            response = requests.get(
                f"https://en.wikipedia.org/w/index.php?search={wiki_query}",
                headers=headers
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check if we've been redirected to a specific page
            content = soup.select_one('#mw-content-text')
            if content:
                # Extract plot section which often mentions characters
                plot_section = soup.find(id='Plot') or soup.find(id='Plot_summary')
                if plot_section:
                    # Get the content following the Plot heading
                    plot_content = []
                    current = plot_section.parent.next_sibling
                    while current and current.name != 'h2':
                        if current.name == 'p':
                            plot_content.append(current.get_text())
                        current = current.next_sibling
                    
                    if plot_content:
                        search_results.append(' '.join(plot_content))
                else:
                    # If no Plot section, just get the main content
                    paragraphs = content.select('p')
                    for p in paragraphs[:5]:
                        search_results.append(p.get_text(strip=True))
        
        # Limit individual search results to avoid overwhelming the model
        limited_search_results = []
        max_words_per_result = 150  # Limit each search result to 150 words
        max_total_words = 800  # Limit total context to 800 words
        
        total_words = 0
        for result in search_results:
            # Split the result into words and limit it
            words = result.split()
            if len(words) > max_words_per_result:
                words = words[:max_words_per_result]
            
            # Check if adding this result would exceed total word limit
            if total_words + len(words) > max_total_words:
                # Only add partial result to stay within limit
                remaining_words = max_total_words - total_words
                if remaining_words > 0:
                    words = words[:remaining_words]
                    limited_search_results.append(" ".join(words))
                break
            
            limited_search_results.append(" ".join(words))
            total_words += len(words)
        
        # Combine limited results into a single text
        combined_text = "\n\n".join(limited_search_results)
        
        # If we have search results, use LLM to extract the protagonist
        if combined_text:
            # Prompt for the AI to extract protagonist information
            prompt = f"""
            Based on the following search results about the book "{book_title}", 
            identify who the protagonist or main character is.
            Only return the name of the protagonist, NOTHING ELSE.
            If you cannot determine the protagonist, return "unknown"
            
            Search results:
            {combined_text}
            """
            
            # Make the API call to LLM

            response = await async_openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": f"{check_if_have_to_include_no_think_token()} You are a helpful assistant that extracts book protagonist information from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # Extract and return the protagonist information
            raw_protagonist_info = response.choices[0].message.content.strip()
            cleaned_protagonist_info = clean_thinking_tags(raw_protagonist_info)

            return cleaned_protagonist_info.lower()
        else:
            return "No search results found for this book."
            
    except requests.exceptions.RequestException as e:
        return f"Error making search request: {str(e)}"
    except Exception as e:
        print(e)
        traceback.print_exc()
        return f"An error occurred: {str(e)}"
    
async def find_book_protagonist(book_title, async_openai_client: AsyncOpenAI, model_name):
    protagonist = "unknown"
    for method in ["google", "wikipedia", "bing", "goodreads", "duckduckgo"]:
        result = await find_book_protagonist_using_search_engine_and_llm(book_title, async_openai_client, model_name, method)
        if result != "No search results found for this book.":
            protagonist = result
            break
    return protagonist
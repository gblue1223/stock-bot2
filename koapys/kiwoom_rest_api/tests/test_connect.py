import httpx
import asyncio
import inspect

async def check_httpx():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get('https://httpbin.org/get')
            response.raise_for_status()

            print(f"Status Code: {response.status_code}")
            print(f"Type of response: {type(response)}")
            json_method = getattr(response, 'json', None)
            text_method = getattr(response, 'text', None)
            print(f"Type of response.json: {type(json_method)}")
            print(f"Type of response.text: {type(text_method)}")
            print(f"Is response.json a coroutine func? {inspect.iscoroutinefunction(response.json)}")
            print(f"Is response.text a coroutine func? False")

            # 실제 호출 시도
            try:
                json_data = response.json()
                print("JSON Data:", json_data)
            except TypeError as e:
                print(f"Processing response.json() failed: {e}")
            except Exception as e:
                print(f"Failed to decode JSON: {e}")

            try:
                text_data = response.text
                print("Text Data:", text_data)
            except TypeError as e:
                print(f"Processing response.text failed: {e}")
            except Exception as e:
                print(f"Failed to access text: {e}")

        except httpx.RequestError as exc:
            print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
        except httpx.HTTPStatusError as exc:
            print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}: {exc.response.text}")

if __name__ == "__main__":
    asyncio.run(check_httpx())
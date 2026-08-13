import json
import requests
import sys
def test_prediction(seq="MKTLLILAVLLLCNNSAGSLGAPQP"):  # Sample sequence
    url = "http://localhost:8001/predict"
    payload = {"sequence": seq}

    print(f"Sending request to {url} with sequence length {len(seq)}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print("Success!")
        print(f"Predicted Stability: {result['stability']}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "MKTLLILAVLLLCNNSAGSLGAPQP"
    test_prediction(seq)

# Movie Recommendation API using Collaborative Based Filtering



## Technologies Used
- **[FastAPI](https://fastapi.tiangolo.com/)**
- **[Surprise](https://surprise.readthedocs.io/en/stable/)**

---

## Steps to run
1. Make a virtual environment with python 3.10.2
    ```bash
    python3.10.2 -m venv {env_name}
    ```

2. Run virtual env and install dependencies
    ```bash
    ./{env_name}/Scripts/activate // for windows
    source {env_name}/bin/activate // for linux

    pip install -r requirements.txt
    ```

3. Run FastAPI server
    ```bash
    uvicorn app:app
    ```

4. Go to http://localhost:8000/docs to test the API.



---

## How It Works


**Collaborative Based Filtering** is a type of recommendation generation system that uses rating provided by other users to generate recommendations (user-user). The rating matrix (rating of every user for every movie) is in reality very sparse. Due to this reason, it is diffucult to find similarity between users. This sparsity can be resolved using Matrix Factorization methods. The sparse matrix is fit to the product of two rectangular matrix (user factor and movie factor) using gradient descent. The product of those two matrices is a filled rating matrix with predicted rating for every movie by every user. After that, to recommend a movie for a user, top n rated movie which the user hasn't already seen can be selected. 


## Outputs

- Recommendations for user 1

![The Lion King](/Screenshots/User1.png)

- Recommendations for user 1

![Space Aliens](/Screenshots/User2.png)


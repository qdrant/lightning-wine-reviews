# wine_reviews_qdrant app

This ⚡ [Lightning app](lightning.ai) ⚡ was generated automatically with:

```bash
lightning init app wine_reviews_qdrant
```

## To run wine_reviews_qdrant

First, install wine_reviews_qdrant (warning: this app has not been officially approved on the lightning gallery):

```bash
lightning install app https://github.com/qdrant/lightning-wine-reviews
```

The application uses Kaggle [wine-reviews dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews)
and handles the download process. However, you need to have Kaggle package configured
with our own credentials: https://www.kaggle.com/docs/api#authentication

Once the app is installed, run it locally with:

```bash
lightning run app wine_reviews_qdrant/app.py
```

On the first launch, a whole dataset will be downloaded and loaded into Qdrant.
That may take a while, so please be careful.

Run it on the [lightning cloud](lightning.ai) with:

```bash
lightning run app wine_reviews_qdrant/app.py --cloud
```

**This is important to provide the Kaggle credentials. For the cloud deployment
the file might not be an option, so they might be overwritten with environmental
variables like that:**

```bash
lightning run app wine_reviews_qdrant/app.py --cloud --env KAGGLE_USERNAME=kaggle_username --env KAGGLE_KEY=kaggle_key
```

## to test and link

Run flake to make sure all your styling is consistent (it keeps your team from going insane)

```bash
flake8 .
```

To test, follow the README.md instructions in the tests folder.

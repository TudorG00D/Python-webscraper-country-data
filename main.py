import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


class CountryData:
    # URL containing the country data to scrape
    URL = "https://www.scrapethissite.com/pages/simple/"

    def __init__(self, database_url="sqlite:///database.db"):
        # Store the database connection and initialize the data containers
        self.engine = create_engine(database_url)
        self.df = None
        self.results = {}

    def scrape(self):
        # Send an HTTP request and parse the page's HTML
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(self.URL, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all country entries on the page
        countries = soup.find_all("div", class_="col-md-4 country")

        # Store the scraped country data as a list of dictionaries
        rows = []
        for country in countries:
            name = country.find("h3", class_="country-name")
            capital = country.find("span", class_="country-capital")
            population = country.find("span", class_="country-population")
            area = country.find("span", class_="country-area")

            rows.append({
                "Country": name.get_text(strip=True) if name else None,
                "Capital": capital.get_text(strip=True) if capital else None,
                "Population": population.get_text(strip=True) if population else None,
                "Area": area.get_text(strip=True) if area else None,
            })

        # Create a pandas dataframe from the scraped data
        self.df = pd.DataFrame(rows)
        return self.df

    def clean_data(self):
        # Ensure the data has been scraped before cleaning it
        if self.df is None:
            raise RuntimeError("Run scrape() before clean_data().")

        # Remove incomplete and duplicate rows, then make a working copy
        df = self.df.dropna().drop_duplicates(subset=["Country"]).copy()

        # Remove leading or trailing whitespace from text fields
        for column in ["Country", "Capital"]:
            df[column] = df[column].str.strip()

        # Convert population and area values to numbers
        for column in ["Population", "Area"]:
            df[column] = pd.to_numeric(
                df[column].astype(str).str.strip(),
                errors="coerce",
            )

        # Remove rows where numeric conversion failed and reset the index
        df = df.dropna(subset=["Population", "Area"]).reset_index(drop=True)

        # Calculate population density for each country
        df["Density"] = df["Population"] / df["Area"]

        self.df = df
        return self.df

    def analyze(self):
        # Ensure the data has been cleaned before analysis
        if self.df is None:
            raise RuntimeError("Run clean_data() before analyze().")

        df = self.df

        # Find the seven most populous countries whose names start with B
        a_df = (
            df[df["Country"].str.upper().str.startswith("B", na=False)]
            .sort_values("Population", ascending=False)
            .head(7)
            .reset_index(drop=True)
        )

        # Find countries with an area of at least one million square kilometers
        b_df = (
            df[df["Area"] >= 1_000_000]
            .sort_values("Area", ascending=False)
            .reset_index(drop=True)
        )

        # Combine the two results, sort by area, remove duplicate countries,
        # and keep the seven countries with the largest areas
        con_df = (
            pd.concat([a_df, b_df], ignore_index=True)
            .sort_values("Area", ascending=False)
            .drop_duplicates(subset="Country")
            .head(7)
            .reset_index(drop=True)
        )

        # Find the five countries with the highest population density
        fiv_df = (
            df.nlargest(5, "Density")
            .reset_index(drop=True)
        )

        # Find the seven least-populated countries with a population above one
        small_df = (
            df[df["Population"] > 1][["Country", "Population"]]
            .nsmallest(7, "Population")
            .reset_index(drop=True)
        )

        # Keep the analysis dataframes together so other methods can use them
        self.results = {
            "countries_type_2": a_df,
            "countries_type_3": b_df,
            "countries_type_4": con_df,
            "countries_type_5": small_df,
            "countries_type_6": fiv_df,
        }
        return self.results

    def save_to_database(self):
        # Ensure the data and analysis results are ready to be saved
        if self.df is None or not self.results:
            raise RuntimeError("Run clean_data() and analyze() before saving.")

        # Combine the main dataframe and analysis dataframes for storage
        tables = {"countries_type_1": self.df, **self.results}

        # Save each dataframe to a SQLite table
        for table_name, frame in tables.items():
            frame.to_sql(
                table_name,
                con=self.engine,
                if_exists="replace",
                index=False,
            )

    def plot(self):
        # Ensure the data and analysis results are ready for plotting
        if self.df is None or not self.results:
            raise RuntimeError("Run clean_data() and analyze() before plotting.")

        small_df = self.results["countries_type_5"]
        con_df = self.results["countries_type_4"]
        fiv_df = self.results["countries_type_6"]

        # Create four plots in a two-by-two layout
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Plot the least-populated countries
        sns.barplot(data=small_df, x="Country", y="Population", ax=axes[0, 0])
        axes[0, 0].set_title("Least Populated Countries and Territories")
        axes[0, 0].set_xlabel("COUNTRY")
        axes[0, 0].set_ylabel("POPULATION")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))

        # Plot the areas of the countries in con_df
        con_df.plot(
            x="Country",
            y="Area",
            kind="bar",
            color="purple",
            ax=axes[0, 1],
            legend=False,
        )
        axes[0, 1].set_title("Area of Countries")
        axes[0, 1].set_xlabel("COUNTRY")
        axes[0, 1].set_ylabel("AREA")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Plot the five most densely populated countries
        fiv_df.plot(
            x="Country",
            y="Density",
            kind="barh",
            color="indigo",
            ax=axes[1, 0],
            legend=False,
        )
        axes[1, 0].set_title("Most Densely Populated")
        axes[1, 0].set_xlabel("DENSITY")
        axes[1, 0].set_ylabel("COUNTRY")

        # Plot country area against population
        self.df.plot(
            x="Area",
            y="Population",
            kind="scatter",
            c="Area",
            cmap="plasma",
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Area vs Population")
        axes[1, 1].set_xlabel("AREA")
        axes[1, 1].set_ylabel("POPULATION")

        # Adjust spacing and display the plots
        plt.tight_layout()
        plt.show()

    def run(self):
        # Run the complete workflow in order
        self.scrape()
        self.clean_data()
        self.analyze()
        self.save_to_database()
        self.plot()


# Create a CountryData object and run the workflow
if __name__ == "__main__":
    country_data = CountryData()
    country_data.run()
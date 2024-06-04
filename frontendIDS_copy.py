import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv("plastic-pollution-by-country-2024.csv")
main_arr = df.values

country_column = df.country
country_df = pd.DataFrame({
    'Column1' : country_column,
    'Column2' : country_column
})

country_array = country_df.values

country_dict = {}
for row in country_array:
    country_dict[row[0]] = row[1]

def getCountryStatus(country):
    for row in main_arr:
        if row[0] == country:
            MWI_level = f"{set_colour(row[2])}"
            plastic_consump = f"Total Plastic Consumption (tons) : {row[3]}"
            plastic_mismanaged = f"Total Plastic Mismanaged (tons) : {row[4]}"
            plastic_consump_capita = f"Plastic Consumption per Capita (tons): {row[5]}"
            exported_plastic = f"Exported Plastic Waste (tons) : {row[6]}"
            imported_plastic = f"Imported Plastic Waste (tons) : {row[7]}"
            plastic_waterways = f"Plastic Waste Released into Waterways (tons): {row[8]}"
            plastic_chemical = f"Plastic Waste from Chemical Additives (tons): {row[9]}"

            status = [MWI_level, plastic_consump, plastic_mismanaged, plastic_consump_capita, exported_plastic, imported_plastic, plastic_waterways, plastic_chemical]

            return status

def set_colour(MWI):
    if MWI == "Very High":
        return '<span style="color: red;">Very High</span>'
    elif MWI == "High":
        return '<span style="color: orange;">High</span>'
    elif MWI == "Medium":
        return '<span style="color: yellow;">Medium</span>'
    elif MWI == "Low":
        return '<span style="color: green;">Low</span>'
    elif MWI == "Very Low":
        return '<span style="color: darkgreen;">Very Low</span>'
    else:
        return MWI

import joblib
model = joblib.load("C://Users//bottl//Downloads//model.joblib")
scaler = joblib.load("C://Users//bottl//Downloads//scaler.joblib")
encoder = joblib.load("C://Users//bottl//Downloads//encoder.joblib")
pca = joblib.load("C://Users//bottl//Downloads//pca.joblib")






# Shiny Frontend
from shiny import App, render, ui
from shinyswatch import theme

app_ui = ui.page_fluid(
    theme.superhero(),
    ui.tags.head(
        ui.tags.style(
            """
            .MWI-large-text {
                font-size: 200px;
            }
            .prediction-header-text {
                font-size: 50px;
            }
            """
        )
    ),
    ui.h1("Plastic Pollution Dashboard", style="text-align: center; margin-top: 20px;"),
    ui.h3("Plastic Pollution by Country", style="text-align: left; margin-top: 30px; margin-bottom: 30px"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_select("dropdown", "Choose a country:", country_dict)
        ),
        ui.panel_main(
            ui.div(
                ui.output_ui("MWI_header_text"),
                ui.output_ui("MWI_Level"),
                style="display: inline;"
            ),
            ui.output_text("plastic_consump"),
            ui.output_text("plastic_mismanaged"),
            ui.output_text("plastic_consump_capita"),
            ui.output_text("exported_plastic"),
            ui.output_text("imported_plastic"),
            ui.output_text("plastic_waterways"),
            ui.output_text("plastic_chemical")
        )
    ),
    ui.h3("Global Infographic", style="text-align: left; margin-top: 30px; margin-bottom: 30px"),
    ui.row(
        ui.column(4, ui.output_plot("pie_plot")),
        ui.column(4, ui.output_plot("bar_plot_highest_plastic_consumption")),
        ui.column(4, ui.output_plot("bar_plot_highest_mismanaged_waste"))
    ),
    ui.h3("MWI Level Predictor", style="text-align: left; margin-top: 30px; margin-bottom: 30px"),
    ui.navset_card_pill(
            ui.nav_panel(
                "Slider",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.input_slider("plastic_consumption_slider", "Total Plastic Consumption (tons)", 0, 10000000, 5000000),
                        ui.input_slider("mismanaged_waste_slider", "Mismanaged Waste (tons)", 0, 1000000, 500000),
                        ui.input_slider("plastic_consumption_per_capita_slider", "Plastic Consumption per Capita (tons)", 0, 50, 25),
                        ui.input_slider("exported_slider", "Exported Waste (tons)", 0, 100000, 50000),
                        ui.input_slider("imported_slider", "Imported Waste (tons)", 0, 1000000, 500000),
                        ui.input_slider("waterways_released_slider", "Waste Released into Waterways (tons)", 0, 100000, 50000),
                        ui.input_slider("chemical_additives_slider", "Waste from Chemical Additives (tons)", 0, 10000, 5000)
                    ),
                    ui.panel_main(
                        ui.div(
                            ui.output_ui("prediction_slider_header_text", class_ = "prediction-header-text"),
                            ui.output_ui("prediction_slider", class_ = "MWI-large-text"),
                            style = "display: grid; grid-template-rows: auto; align-content: center; height: 100%; width: 100%;"
                        )
                    )
                )
            ),
            ui.nav_panel(
                "Manual",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.input_numeric("plastic_consumption", "Total Plastic Consumption (tons)", 0, min=0, max=10000000),
                        ui.input_numeric("mismanaged_waste", "Mismanaged Waste (tons)", 0, min=0, max=1000000),
                        ui.input_numeric("plastic_consumption_per_capita", "Plastic Consumption per Capita (tons)", 0, min=0, max=50),
                        ui.input_numeric("exported", "Exported Waste (tons)", 0, min=0, max=100000),
                        ui.input_numeric("imported", "Imported Waste (tons)", 0, min=0, max=1000000),
                        ui.input_numeric("waterways_released", "Waste Released into Waterways (tons)", 0, min=0, max=100000),
                        ui.input_numeric("chemical_additives", "Waste from Chemical Additives (tons)", 0, min=0, max=10000)
                    ),
                    ui.panel_main(
                        ui.div(
                            ui.output_ui("prediction_header_text", class_ = "prediction-header-text"),
                            ui.output_ui("prediction_manual", class_ = "MWI-large-text"),
                            style = "display: grid; grid-template-rows: auto; align-content: center; height: 100%; width: 100%;"
                        )
                    )
                )
            )
        
    )
)

def server(input, output, session):    
    @output
    @render.ui
    def MWI_Level():
        status = getCountryStatus(input.dropdown())
        mwi = status[0]
        return ui.HTML(mwi)
    
    @output
    @render.text
    def plastic_consump():
        status = getCountryStatus(input.dropdown())
        return status[1]
    
    @output
    @render.text
    def plastic_mismanaged():
        status = getCountryStatus(input.dropdown())
        return status[2]
    
    @output
    @render.text
    def plastic_consump_capita():
        status = getCountryStatus(input.dropdown())
        return status[3]
    
    @output
    @render.text
    def exported_plastic():
        status = getCountryStatus(input.dropdown())
        return status[4]
    
    @output
    @render.text
    def imported_plastic():
        status = getCountryStatus(input.dropdown())
        return status[5]
    
    @output
    @render.text
    def plastic_waterways():
        status = getCountryStatus(input.dropdown())
        return status[6]
    
    @output
    @render.text
    def plastic_chemical():
        result = getCountryStatus(input.dropdown())
        return result[7]
    
    @output
    @render.plot
    def pie_plot():
        sns.set_style("whitegrid")
        plt.pie(df['PlasticPollutionMWILevel'].value_counts(), labels = df['PlasticPollutionMWILevel'].unique())
        plt.title("Distribution of MWI Levels")

    @output
    @render.plot
    def bar_plot_highest_plastic_consumption():
        consumption_top10 = df.sort_values(by='PlasticPollutionTotalPlasticConsumption_intonsofplasticwaste', ascending=False).head(10)

        sns.barplot(consumption_top10, x = 'PlasticPollutionTotalPlasticConsumption_intonsofplasticwaste', y = 'country', orient = 'h')
        plt.title('Top 10 Countries with Highest Plastic Consumption')

    @output
    @render.plot
    def bar_plot_highest_mismanaged_waste():
        mismanaged_top10 = df.sort_values(by='PlasticPollutionMismanagedWaste2023_expected_tons', ascending = False).head(10)

        sns.barplot(mismanaged_top10, x = 'PlasticPollutionMismanagedWaste2023_expected_tons', y = 'country', orient='h')
        plt.title('Top 10 Countries with Highest Mismanaged Waste')

    @output
    @render.ui
    def prediction_slider():
        inputs = [[
            input.plastic_consumption_slider(),
            input.mismanaged_waste_slider(),
            input.plastic_consumption_per_capita_slider(), 
            input.exported_slider(), 
            input.imported_slider(), 
            input.waterways_released_slider(), 
            input.chemical_additives_slider()
            ]]
        
        inputs = pca.transform(inputs)
        inputs = scaler.transform(inputs)

        result = model.predict(inputs)
        result = encoder.inverse_transform(result)

        mwi = set_colour(result[0])
        return ui.HTML(mwi)

    @output
    @render.ui
    def prediction_manual():
        inputs = [[
            input.plastic_consumption(), 
            input.mismanaged_waste(), 
            input.plastic_consumption_per_capita(), 
            input.exported(), 
            input.imported(), 
            input.waterways_released(), 
            input.chemical_additives()
            ]]
        
        try:
            inputs = pca.transform(inputs)
        except Exception:
            return ui.output_ui("error_handler")
            
        inputs = scaler.transform(inputs)

        result = model.predict(inputs)
        result = encoder.inverse_transform(result)

        mwi = set_colour(result[0])
        return ui.HTML(mwi)
    
    @output
    @render.ui
    def prediction_header_text():
        return "Predicted MWI Level: "
    
    @output
    @render.ui
    def prediction_slider_header_text():
        return "Predicted MWI Level: "
    
    @output
    @render.ui
    def MWI_header_text():
        return ui.HTML('<span display: inline;">MWI Level: </span>')
    
    @output
    @render.ui
    def error_handler():
        return "   "
        
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
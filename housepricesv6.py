# https://stackoverflow.com/questions/65507374/plotting-a-geopandas-dataframe-using-plotly
# https://plotly.com/python/choropleth-maps/#using-geopandas-data-frames
# https://mapshaper.org/
# https://github.com/mbloch/mapshaper/issues/432#issuecomment-675775465
# FOR GEOJSON FILE:
# Download geojson file from gov geoportal
# Need to reduced size of file, go to mapshaper.org
# In mapshaper go to console and type in  -o gj2008
# use downloaded json file in setup

# TO DO:


# Set up imports:
import json
import dash
import statistics
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash_bootstrap_components.themes import UNITED
from dash_bootstrap_templates import load_figure_template
import numpy as np
import dash_auth

# Start Dash app -----------------------------------------------------------
# The external stylesheet is a bootstrap template
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
server = app.server

load_figure_template("UNITED")

#List of colours to use in design:

#colours = [(0, "#fdcf58"),(0.2, "#f27d0c"), (0.4, "#d9700a"), (0.6, "#c16409" ), (0.8, '#a95708') ,(1,"#800909" )]

# set up dfs -----------------------------------------------------------------
df = pd.read_csv(r"LAlookup.csv")
earningdf = pd.read_csv(r"Medianincome.csv")
housedf = pd.read_csv(r"Medianhouseprice.csv")


#set up options for year dropdown ----------------------------------------------------

years_list = list(earningdf)
years_list.remove("LAD21NM")

years_options_list = []

for y in years_list:
    years_dict = dict()
    years_dict["label"] = y
    years_dict["value"] = y
    years_options_list.append(years_dict)

#Create a function to create a dataframe that uses a selected region and year --------------------------------------------------

#Have to convert to numeric as imported as string
for x in years_list:
    earningdf[x] = pd.to_numeric(earningdf[x], errors='coerce')

def yearregiondataframe(region, year):
    #filter imported dataframes to selected year and region
    newdf = df[df["Region"] == region]
    earningdfbyyear = earningdf [["LAD21NM", year]]
    housedfbyyear = housedf [["LAD21NM", str(year)]]

    #merge the three dataframes
    df1 = pd.merge(
        left=newdf,
        right=earningdfbyyear,
        left_on='LAD21NM',
        right_on='LAD21NM',
        how='left')

    df2 = pd.merge(left=df1,
                   right=housedfbyyear,
                   left_on='LAD21NM',
                   right_on='LAD21NM',
                   how='left'
                   )

    #create a new column for the ratio of house prices to earnings
    earningcolname = (year + "_x")
    housecolname = (year + "_y")

    def ratio_row(row):
        x = row[housecolname]
        y = row[earningcolname]
        z = x/y
        return z

    df2['Ratio'] = df2.apply(lambda row: ratio_row(row), axis=1)

    #rename columns
    df2.rename(columns = {"LAD21NM": "Local Authority"}, inplace = True)

    #create column for percentile rank
    df2['pct_rank'] = (df2['Ratio'].rank(pct=True))*100

    # df2['pct_rank'] = df2['pct_rank'].replace(np.nan, "")

    df2 = df2.round(1)

    #return final dataframe
    return df2

# Set up options for region dropdown ---------------------------------------------------
regions = list(df.Region.unique())
regions = [item for item in regions if not (pd.isnull(item)) == True]

region_options_list = []

for region in regions:
    region_dict = dict()
    region_dict["label"] = str(region)
    region_dict["value"] = str(region)
    region_options_list.append(region_dict)

# Set up heading content and style ------------------------------------------------------------------

title = html.Div(
    [
        html.H2("Average house prices to earnings in England", style = {'color': ' #f27d0c'}),
        html.Hr()
    ],
    style = {"padding": "0.5rem 0.5rem 0.5rem"}
)

#----------------set up filters-------------------------------------------

region_filter = html.Div(
    [html.P(
        "selected region: ", className="lead"
    ),
        html.Div(dcc.Dropdown(id='region_picker', options=region_options_list, value="England", clearable=False), style={
            'width': '80%'
        }),
        html.Br(),
    ],
    style = {"padding": "0.5rem 0.5rem 0.5rem"}
)

year_filter = html.Div(
    [html.P(
        "selected year: ", className="lead"
    ),
        html.Div(dcc.Dropdown(id='year_picker', options=years_options_list, value="2021", clearable = False),
                 style={
            'width': '60%', "float":"left"}),
        html.Br(),
    ],
    style = {"padding": "0.5rem 0.5rem 0.5rem" }
)

#----------------set up region title and style

region_title = html.Div([html.Br(),
                         html.Br(),
                        html.H2(id = "click-output2", style={"float":"left", "margin-left": "30px"})],
                        style = {"position":"relative"})

#-----------------set up sidebar content and style


breakdownbar = html.Div(
    [
        html.P(id = "click-output3", className = "lead"),
        html.Hr(id = "line_break"),
        html.P(id = "click-output", className= "lead"),
        html.Div(id = "barchartcontainer", children =[
        dcc.Graph(id="barchart", style= {'width': '38vw', 'height': '40vh'}, config= {'displayModeBar': False})]),
        html.P("Local authorities with missing data have been excluded", style = {'fontSize': 15, 'position': 'fixed', 'bottom': 1})
    ],
    style = {"padding": "2rem 2rem 2rem", "margin-right": "15px"}
)

#---------------set up choropleth graph content and style

graph_bar = html.Div ([
    html.Div(dcc.Graph(id='graph',
                       style = {'width': '50vw', 'height': '70vh', "margin-left": "5px"}, config= {'displayModeBar': False}),
             style={'width': '100%', 'position':'relative'})
])

# set up app layout ------------------------------------------------------------------------------

app.layout = dbc.Container(children=[
    dbc.Row([
        dbc.Col(title, width = 12)
    ]),

    dbc.Row([
        dbc.Col(region_filter, width = 2),
        dbc.Col(year_filter, width = 2),
        dbc.Col(width = 3),
        dbc.Col(region_title, width = 5)
    ]),

    dbc.Row(
        [
            dbc.Col(graph_bar, width = 7),
            dbc.Col(breakdownbar,
                    width=5)
        ])
], fluid=True)

# choropleth graph creation and callback based on region -----------------------------------------------------------

@app.callback(
    Output("graph", "figure")
#Maybe a way to fix it is to use the STATE instead of input to get the current selectedData, then check it or 
#reset it. Honestly not too sure though
#https://dash.plotly.com/advanced-callbacks
#https://community.plotly.com/t/callback-origin-and-reset-clickdata-on-python-dash/29805
#https://stackoverflow.com/questions/69426319/scattermapbox-clickdata-callback-is-not-triggerd-when-unselect-item-in-map
# https://community.plotly.com/t/what-is-the-difference-between-input-and-state/35219
#
    ,
    Input("region_picker", "value"),
    Input("year_picker", "value")
)

def display_choropleth(region, year):

    #Use data frame function to create dataframe for selected region and year
    choropleth_df = yearregiondataframe(region, year)

    # Set up json file
    with open(r"LA_map.json") as json_file:
        geojson = json.load(json_file)

    # create figure
    fig = px.choropleth_mapbox(choropleth_df, geojson=geojson, color=choropleth_df["Ratio"],
                               locations=choropleth_df["Local Authority"], featureidkey="properties.LAD21NM",
                               color_continuous_scale= "YlOrRd",
                               hover_data= ["Local Authority"], opacity=0.7,
                               )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(autosize=False, margin={"r": 0, "t": 0, "l": 0, "b": 0, "autoexpand": True, "pad": 1}, width=900)



    #Centres to use in mapbox_center zoom
    centre_lat_long = {"England": {"lat": 52.567004, "lon": -1.180832},
                       "East of England": {"lat": 52.238061, "lon": 0.571506},
                       "West Midlands": {"lat": 52.523157, "lon": -2.189539},
                       "East Midlands": {"lat": 52.871699, "lon": -0.679063},
                       "South East": {"lat": 51.281418, "lon": -0.339193},
                       "South West": {"lat": 51.065737, "lon": -2.890518},
                       "North West": {"lat": 54.046005, "lon": -2.562678},
                       "North East": {"lat": 55.088114, "lon": -1.655882},
                       "London": {"lat": 51.507204, "lon": -0.119856},
                       "Yorkshire and The Humber": {"lat": 53.920193, "lon": -1.097800}
                       }

    #To get zoom right when different regions are selected
    non_london_zoom = 7
    zoom_for_mapbox =  {"England": 5.5,
                        "East of England": non_london_zoom,
                        "West Midlands": non_london_zoom,
                        "East Midlands":non_london_zoom ,
                        "South East": non_london_zoom,
                        "South West": 6.5,
                        "North West": non_london_zoom,
                        "North East": non_london_zoom,
                        "London": 9,
                        "Yorkshire and The Humber":non_london_zoom
                        }

    fig.update_layout(mapbox_style="carto-positron",
                      mapbox_zoom= zoom_for_mapbox[region],
                      mapbox_center={"lat": centre_lat_long[region]["lat"], "lon": centre_lat_long[region]["lon"]},
                      margin={"r":0,"t":0,"l":0,"b":0},
                      uirevision='constant')

    # fig.update_layout(modebar_remove=['select2d', 'lasso2d'])

    fig.update_layout(coloraxis_colorbar=dict(
        # title="Vacant homes to homeless ratio",
        yanchor="top", y=1,
        xanchor="left", x=1,
        dtick=5))

    fig.update_layout(clickmode='event+select')

    return fig


#breakdown creation and callback based on region -----------------------------------------------------------

@app.callback(
    Output('click-output', 'children'),
    Input('graph', 'selectedData'),
    Input("region_picker", "value"),
    Input("year_picker", "value"))


def display_selected_data(selectedData, region, year):
    #displays message if no LAs are clicked on
    if selectedData is None:
        noselectmessage = str("Click on a local authority to see further information")
        return noselectmessage
    else:
        displaydatadf = yearregiondataframe(region, year)

        name_of_LA = f'{selectedData["points"][0]["location"]}'

        #Need to check if the region has changed since the last local
        #authority was selected 
        tocheckifinregion = displaydatadf["Local Authority"].tolist()

        if name_of_LA not in tocheckifinregion:

            noselectmessage = str("Click on a local authority to see further information")
            return noselectmessage

        else:

            df_for_selected = displaydatadf[displaydatadf["Local Authority"] == name_of_LA]
            df_for_selected = df_for_selected.reset_index()
            earningcolname = (year + "_x")
            housecolname = (year + "_y")
            house_prices_for_LA = int(df_for_selected[housecolname].loc[0])
            earnings_for_LA = int(df_for_selected[earningcolname].loc[0])
            house_prices_for_LA_formatted = "{:,.0f}".format(house_prices_for_LA)
            earnings_for_LA_formatted = "{:,.0f}".format(earnings_for_LA)

            return f'In {year}, the median salary in {name_of_LA} was £{earnings_for_LA_formatted}.' \
                f' \n At the same time, the average house price was £{house_prices_for_LA_formatted}.'



#Name of LA and it's region ------------------------------------

@app.callback(
    Output('click-output2', 'children'),
    Input('graph', 'selectedData'))


def display_selected_data2(selectedData):
    if selectedData is None:
        return f' '
    else:
        name_of_LA = f'{selectedData["points"][0]["location"]}'
        df_for_selected = df[df["LAD21NM"] == name_of_LA]
        df_for_selected = df_for_selected.reset_index()
        region_for_LA = f'{df_for_selected["Region"].loc[0]}'

        return f'{name_of_LA}, {region_for_LA}'


#Percentile rank for LA ----------------------------------------------

@app.callback(
    Output('click-output3', 'children'),
    Input('graph', 'selectedData'),
    Input("region_picker", "value"),
    Input("year_picker", "value"))


def display_selected_data3(selectedData, region, year):
    if selectedData is None:
        return f''
    else:
        displaydatadf = yearregiondataframe(region, year)
        name_of_LA = f'{selectedData["points"][0]["location"]}'

        #Check if selected data is in the displaydatadf, if not then needs to be updated otherwise error.
        tocheckifinregion = displaydatadf["Local Authority"].tolist()

        if name_of_LA not in tocheckifinregion:

            noselectmessage = str("Click on a local authority to see further information")

            return f''

        else:

            df_for_selected = displaydatadf[displaydatadf["Local Authority"] == name_of_LA]
            df_for_selected = df_for_selected.reset_index()
            name_of_region = str(region)
            pct_rank_value = int(df_for_selected["pct_rank"].loc[0])

            return f'Percentile rank in {name_of_region}: {pct_rank_value}'


#App callback to remove grey line under Rank if no LA selected------------------------------------


@app.callback(
    Output('line_break', 'hidden'),
    Input('graph', 'selectedData'))

def remove_line(selectedData):
    if selectedData is None:
        return True
    else:
        return False


# bar chart creation and callback based on region -----------------------------------------------------------

@app.callback(
    Output("barchart", "figure"),
    Input("region_picker", "value"),
    Input("year_picker", "value"),
    Input('graph', 'selectedData')
)

def createbarchart(region, year, selectedData):
    if selectedData is None:
        return {}
    elif year == "2010":
        return {}
    else:
        x = yearregiondataframe(region,year)

        name_of_LA = f'{selectedData["points"][0]["location"]}'

        tocheckifinregion = x["Local Authority"].tolist()

        if name_of_LA not in tocheckifinregion:
            return {}
        
        else:

            def geo_mean(iterable):
                a = np.array(iterable)
                return a.prod()**(1.0/len(a))

            x.rename(columns = {"Local Authority": "LAD21NM"}, inplace = True)

            earningdfbyyear = earningdf [["LAD21NM", "2010"]]
            housedfbyyear = housedf [["LAD21NM", "2010"]]

            df1_1 = pd.merge(
                left=x,
                right=earningdfbyyear,
                left_on='LAD21NM',
                right_on='LAD21NM',
                how='left')

            df2 = pd.merge(left=df1_1,
                        right=housedfbyyear,
                        left_on='LAD21NM',
                        right_on='LAD21NM',
                        how='left'
                        )

            earningcolname = (year + "_x")
            housecolname = (year + "_y")

            #--------------to figure out average regional % increase
            avg_earninglist_selectedyear = df2[earningcolname].dropna().tolist()
            mean_earnings_selected_year = statistics.geometric_mean(avg_earninglist_selectedyear)

            avg_houselist_selectedyear = df2[housecolname].dropna().tolist()
            mean_house_selected_year = statistics.geometric_mean(avg_houselist_selectedyear)

            avg_earninglist_2010 = df2["2010_x"].dropna().tolist()
            mean_earnings_2010 = statistics.geometric_mean (avg_earninglist_2010)

            avg_houselist_2010 = df2["2010_y"].dropna().tolist()
            mean_house_2010 = statistics.geometric_mean(avg_houselist_2010)

            region_percent_earning_increase = ((mean_earnings_selected_year - mean_earnings_2010)/mean_earnings_2010) * 100
            region_percent_house_increase = ((mean_house_selected_year - mean_house_2010)/mean_house_2010) * 100

            #----------------------- to check for nan
            #df for selected also used to figure our LA % increase
            df_for_selected = df2[df2["LAD21NM"] == name_of_LA]

            check_for_nan = df_for_selected.isnull().values.any()
            if check_for_nan == True:
                return {}
            else:

                #----------------------- to figure out LA % increases

                LA_housevalue_selected_year = df_for_selected[housecolname].loc[df_for_selected.index[0]]
                LA_housevalue_2010 = df_for_selected["2010_y"].loc[df_for_selected.index[0]]

                LA_earningvalue_selected_year = df_for_selected[earningcolname].loc[df_for_selected.index[0]]
                LA_earningvalue_2010 = df_for_selected["2010_x"].loc[df_for_selected.index[0]]

                LA_percent_earning_increase = ((LA_earningvalue_selected_year - LA_earningvalue_2010)/LA_earningvalue_2010) * 100
                LA_percent_house_increase = ((LA_housevalue_selected_year - LA_housevalue_2010)/LA_housevalue_2010) * 100

                #----------------------make df for graph--------------------------------------------------

                percentage_increase = [LA_percent_earning_increase, LA_percent_house_increase,
                                    region_percent_earning_increase, region_percent_house_increase]

                area =[name_of_LA, name_of_LA, region, region]

                metricname = ["earnings", "house prices", "earnings", "house prices"]

                dataframe_for_comparison_graph = pd.DataFrame(zip(percentage_increase, area, metricname ),
                                                            columns = ["percentage increase", "geography", "metric"])

                fig = px.bar(dataframe_for_comparison_graph, x="metric", y="percentage increase",
                            color="geography", barmode="group")

                fig.update_xaxes(title='')

                fig.update_layout(legend=dict(
                    title=""
                ))

                fig.update_layout(
                    title={
                        'text': "<b> percentage increase since 2010 </b>",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    yaxis_title=None)

                return fig

#--To remove the graph from the page when it's not valid--------------------------------------------------------------------
#Most of the code is identical to above - might be worth making a function and reducing this

@app.callback(
    Output("barchartcontainer", "style"),
    Input("region_picker", "value"),
    Input("year_picker", "value"),
    Input('graph', 'selectedData')
)

def removebarchart(region, year, selectedData):
    if selectedData is None:
        return {'display':'none'}
    elif year == "2010":
        return {'display':'none'}
    else:
        x = yearregiondataframe(region,year)

        name_of_LA = f'{selectedData["points"][0]["location"]}'

        def geo_mean(iterable):
            a = np.array(iterable)
            return a.prod()**(1.0/len(a))

        x.rename(columns = {"Local Authority": "LAD21NM"}, inplace = True)

        earningdfbyyear = earningdf [["LAD21NM", "2010"]]
        housedfbyyear = housedf [["LAD21NM", "2010"]]

        df1_1 = pd.merge(
            left=x,
            right=earningdfbyyear,
            left_on='LAD21NM',
            right_on='LAD21NM',
            how='left')

        df2 = pd.merge(left=df1_1,
                       right=housedfbyyear,
                       left_on='LAD21NM',
                       right_on='LAD21NM',
                       how='left'
                       )

        earningcolname = (year + "_x")
        housecolname = (year + "_y")

        #--------------to figure out average regional % increase
        avg_earninglist_selectedyear = df2[earningcolname].dropna().tolist()
        mean_earnings_selected_year = statistics.geometric_mean(avg_earninglist_selectedyear)

        avg_houselist_selectedyear = df2[housecolname].dropna().tolist()
        mean_house_selected_year = statistics.geometric_mean(avg_houselist_selectedyear)

        avg_earninglist_2010 = df2["2010_x"].dropna().tolist()
        mean_earnings_2010 = statistics.geometric_mean (avg_earninglist_2010)

        avg_houselist_2010 = df2["2010_y"].dropna().tolist()
        mean_house_2010 = statistics.geometric_mean(avg_houselist_2010)

        region_percent_earning_increase = ((mean_earnings_selected_year - mean_earnings_2010)/mean_earnings_2010) * 100
        region_percent_house_increase = ((mean_house_selected_year - mean_house_2010)/mean_house_2010) * 100

        #----------------------- to check for nan
        #df for selected also used to figure our LA % increase
        df_for_selected = df2[df2["LAD21NM"] == name_of_LA]

        check_for_nan = df_for_selected.isnull().values.any()
        if check_for_nan == True:
            return {'display':'none'}
        else:
            return {'display':'block'}

#------The below basically clears the selected Data when nothings selected but it doesn't fix the issue
#with the choropleth map 

@app.callback(Output("graph", "selectedData"),
        Input("region_picker", "value"))

def update(val):
    return None


if __name__ == '__main__':
    app.run_server(debug=False)

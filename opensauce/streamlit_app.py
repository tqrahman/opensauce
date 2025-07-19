from ast import literal_eval
import requests
import string
import json
import requests
from os import path, getcwd
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sns

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# API parameters

papaduck = "OPNSAUCE" # Enter your own papaduck
start_time = 1752901902 # Put start date
end_time = 1846010000 # Put end date
token_timer = datetime.datetime.now().timestamp()
credentials = {
    "username": "uconnteam42@gmail.com", # Enter your own username
    "password": "Papaduck@42"  # Enter your own password
}

def create_api_query(papaduck, start_time, end_time):
    '''
    Creates a query to get data given a papaduck, start time, and end time

    Parameters:
    ------------
    papaduck: the PapaDuck used to collect network data and push it to cloud
    start_time: the desired start time (in epoch) for the data
    end_time: the desired end time (in epoch) for the data

    Returns:
    ---------
    query string
    '''
    
    return "?start="+str(start_time) + "&end=" + str(end_time) + "&papaId=" + papaduck

def load_credentials():
    '''
    Read in the DMS username and password from a json file named 'credentials.json'

    Parameters:
    ------------
    None

    Returns:
    ---------
    json object that contains login information (username and password) for the DMS
    '''

    ## Creating the file path to credentials.json
    cwd = getcwd()
    filename = path.join(cwd, 'credentials.json')

    ## Saving the content from json into credentials
    with open(filename, 'r') as f:
        credentials = json.load(f)
    return credentials

def get_token(credentials):
    '''
    Gets a token from a POST request for an authorized user

    Parameters:
    ------------
    credentials: json object containing login information for DMS 

    Returns:
    ---------
    a json object containing the Bearer token for the authorized user
    '''
    
    # Getting the token
    post = requests.post("https://beta.owldms.com/owl/api/users/authenticate", json=credentials)
    
    # Creating the header token
    global token_header
    token_header = {
        'Authorization':'Bearer '+post.json()["token"]
    }

    global token_timer
    token_timer = datetime.datetime.now().timestamp()

    print("Token: ", token_header)

    # return token_header


create_api_query(papaduck, start_time, end_time)

api_query = "https://beta.owldms.com/owl/api/userdata/getrawdata?start=1746056137&end=1846010000&papaId=OPNSAUCE"

token_header = {'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6IjEwMyIsIm5iZiI6MTc1MjgwOTU1NiwiZXhwIjoxNzUyODk1OTU2LCJpYXQiOjE3NTI4MDk1NTZ9.In0w1EPv6zGKjqi-zeKZpPylVeVrrjDIMl-_AmhLAg0'}

# Function to convert payload string into a dict
def parse_payload(payload):
    parts = payload.split('|')
    result = {}
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            result[key.strip()] = value.strip()
        else:
            result['status'] = part.strip()
    return result

def visualize_grid(grid_size, colored_cells):
    grid = np.full(grid_size, '#FFFFFF', dtype=object)
    for color, x, y in colored_cells:
        if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
            grid[x, y] = color

    fig, ax = plt.subplots(figsize=(10, 14))
    fig.patch.set_facecolor('black')
    rgb_grid = np.array([[mcolors.to_rgb(c) for c in row] for row in grid])
    ax.imshow(rgb_grid, aspect='equal')

    # Add gridlines
    ax.set_xticks(np.arange(-.5, grid_size[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.5, alpha=0.4)

    # Overlay coordinates (A1, B2, ...)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            coord_label = f"{string.ascii_uppercase[i]}{j+1}"
            ax.text(j, i, coord_label, va='center', ha='center', color='gray', fontsize=7, alpha=0.6)

    # Hide default tick marks
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # plt.show()
    st.pyplot(fig)

def coordinate_to_index(coord_str):
    """
    Converts a coordinate like 'T10' to (row_index, col_index)
    'A' corresponds to 0, so 'T' -> 19 and '10' -> 9 (0-based)
    """
    letter_part = coord_str[0].capitalize()  # Ensure single letter is capitalized
    number_part = coord_str[1]

    if not letter_part or not number_part:
        raise ValueError(f"Invalid coordinate format: {coord_str}")

    # Convert letters to row (supports AA, AB, etc.)
    row_index = 0
    for i, char in enumerate(reversed(letter_part)):
        row_index += (string.ascii_uppercase.index(char) + 1) * (26 ** i)
    row_index -= 1  # zero-based

    col_index = int(number_part) - 1  # zero-based

    return row_index, col_index

def fill_in_coordinate(list_of_incoming_message):
    coord_map = {}
    for incoming_message in list_of_incoming_message:
        color, grid_x, grid_y = incoming_message
        x, y = coordinate_to_index([grid_x, grid_y])
        coord_map[(x, y)] = color  # Latest one wins!

    # Convert back to list of (color, x, y)
    colored_cells = [(color, x, y) for (x, y), color in coord_map.items()]
    visualize_grid(grid_size, colored_cells)

def get_api_data():
    try:
        if datetime.datetime.now().timestamp() - token_timer >= 86400:
            global token_header
            token_header = get_token(credentials)
        response = requests.get(api_query, headers=token_header)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching API data: {e}")
        return pd.DataFrame()

# Your coordinates
coords = [
    "F4", "F5", "F6",       "F8", "F9", "F10",        "F12", "F13", "F14",     "F16", "F17", "F18",
    "G4",       "G6",       "G8",       "G10",        "G12",                   "G16",        "G18",
    "H4",       "H6",       "H8", "H9", "H10",        "H12", "H13", "H14",     "H16",        "H18",
    "I4",       "I6",       "I8",                     "I12",                   "I16",        "I18",
    "J4", "J5", "J6",       "J8",                     "J12", "J13", "J14",     "J16",        "J18",
    
    "L4", "L5", "L6",       "L8", "L9", "L10",        "L12",        "L14",     "L16", "L17", "L18",     "L20", "L21", "L22",
    "M4",                   "M8",       "M10",        "M12",        "M14",     "M16",                   "M20", 
    "N4", "N5", "N6",       "N8", "N9", "N10",        "N12",        "N14",     "N16",                   "N20", "N21", "N22",
                "O6",       "O8",       "O10",        "O12",        "O14",     "O16",                   "O20",
    "P4", "P5", "P6",       "P8",       "P10",        "P12", "P13", "P14",     "P16", "P17", "P18",     "P20", "P21", "P22",

                                                                   
                                                                    "R14", "R15", 
                            "T8",                            "S13", "S14", "S15", "S16", 
                            "U8", "U9",                             "T14", "T15",
                                  "V9", "V10", "V11", "V12", "V13", "V14",
                                  "W9", "W10", "W11", "W12", "W13", "W14",
                                        "X10", "X11", "X12", "X13",
                                                                    
]

# Convert to list of (color, x_index, y_index)
color_cells = [["#EBEBEB", c[0], int(c[1:])] for c in coords]
# print(color_cells)
grid_size = (26,26)

# Auto-refresh every 10 seconds
st_autorefresh(interval=10_000, key="api_refresh")

df = get_api_data()
df['payload'] = df['payload'].apply(lambda x: literal_eval(x))
df = pd.concat([df, df['payload'].apply(pd.Series)], axis=1)
df = df.drop(columns=['payload'])
df['createdAt'] = pd.to_datetime(df['createdAt'], format='ISO8601')
df = df.sort_values(by="createdAt")

alert_df = df.loc[df['eventType']== 'alert',].copy()
# print(alert_df['Payload'].head())
alert_df['color_coordinates'] = alert_df['Payload'].str.split('|').apply(lambda x: [x[0], x[1], int(x[2])])

color_cells.extend(alert_df['color_coordinates'].to_list())

sensor_data = df.loc[df['eventType'] == 'bmp180', ].copy()
parsed_df = sensor_data['Payload'].apply(parse_payload).apply(pd.Series)
sensor_data = pd.concat([sensor_data.drop(columns=['Payload']), parsed_df], axis=1)
sensor_data['date'] = sensor_data['createdAt'].dt.date
sensor_data['time'] = sensor_data['createdAt'].dt.time

# health_data = df.loc[df['eventType'] == 'health', ].copy()
# parsed_df = health_data['Payload'].apply(parse_payload).apply(pd.Series)
# health_data = pd.concat([health_data.drop(columns=['Payload']), parsed_df], axis=1)

# status_data = df.loc[df['eventType'] == 'status', ].copy()
# parsed_df = status_data['Payload'].apply(parse_payload).apply(pd.Series)
# status_data = pd.concat([status_data.drop(columns=['Payload']), parsed_df], axis=1)

# Create tabs
tab1, tab2 = st.tabs(["Cluster Art", "🦆 Duck Management System"])
with tab1:

    if not df.empty:
        # visualize_grid(grid_size, colored_cells)
        fill_in_coordinate(color_cells)
    else:
        st.warning("No data returned from API.")

with tab2:
    fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    ax1.plot(sensor_data['createdAt'], sensor_data['T'])
    ax1.set_xlabel('Time (GMT)')
    ax1.set_ylabel('Temperature (C)')
    ax1.set_title('Temperature over Time')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)
    ax2.plot(sensor_data['createdAt'], sensor_data['P'])
    ax2.set_xlabel('Time (GMT)')
    ax2.set_ylabel('Pressure (hPa)')
    ax2.set_title('Pressure over Time')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 5), dpi=150)
    ax3.plot(sensor_data['createdAt'], sensor_data['A']);
    ax3.set_xlabel('Time (GMT)');
    ax3.set_ylabel('Altitude (m)');
    ax3.set_title('Altitude over Time');
    ax3.tick_params(axis='x', rotation=45);
    ax3.grid(True);
    st.pyplot(fig3);
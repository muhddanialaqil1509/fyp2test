import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import datetime
import altair as alt

if not st.session_state:
    st.session_state.chart={"Heart Rate Predicted":[]}

def upload_file():
    body = st.empty()

    with body.container():
        st.title(f"""MISSING HEART RATE DATA PREDICTION USING HEART RATE AND STEPS DATA FROM MI BAND 6""")
        st.subheader(f"This app will predict the heart rate of the Long COVID patients based on heart rate and steps data.")

        st.write(f'## Please upload the 3 csv files mentioned as below.')
        st.write("Samples data can be found on : http://shorturl.at/tJ356")
        st.write("P/S: In order to predict or re-predict, please click submit button again.")

        with st.form(key="form1"):
            path1 = st.file_uploader("UPLOAD ACTIVITY_MINUTE FILE", type={"csv"})
            path2 = st.file_uploader("UPLOAD HEARTRATE_AUTO FILE", type={"csv"})
            path3 = st.file_uploader("UPLOAD USER FILE", type={"csv"})
            submitted = st.form_submit_button()

            if submitted and path1 and path2 and path3:
                activity_min = pd.read_csv(path1)
                heartrate_auto = pd.read_csv(path2)
                user = pd.read_csv(path3)

                try:

                    try:
                        # Checking the missing values
                        activity_min.isnull().sum()
                        heartrate_auto.isnull().sum()
                        user.isnull().sum()

                        steps_df = activity_min
                        hr_df = heartrate_auto
                        user_df = user

                    except:
                        st.error("Data not found.")


                    # Add the necessary number of days to the date value and subtract the corresponding number of hours from the time value
                    def fix_time(row):
                        time = row["time"]
                        hour = int(time.split(":")[0])
                        minute = int(time.split(":")[1])
                        while hour >= 24:
                            row["date"] = pd.to_datetime(row["date"]) + pd.DateOffset(days=1)
                            hour -= 24
                            time = f"{hour:02d}:{minute:02d}"
                            row["time"] = time
                        # Convert the date column to a Timestamp object
                        date = pd.to_datetime(row["date"])
                        # Convert the Timestamp object to a string using the strftime method and the desired date format
                        date_str = date.strftime("%Y-%m-%d")
                        # Combine the date and time columns into a single datetime string
                        datetime_str = date_str + " " + row["time"]
                        # Convert the datetime string to a datetime object
                        datetime = pd.to_datetime(datetime_str)
                        row["datetime"] = datetime
                        # Classify the time of day based on the hour value
                        if datetime.hour >= 6 and datetime.hour < 12:
                            row["time_of_day"] = "morning"
                        elif datetime.hour >= 12 and datetime.hour < 18:
                            row["time_of_day"] = "afternoon"
                        elif datetime.hour >= 18 and datetime.hour < 22:
                            row["time_of_day"] = "evening"
                        else:
                            row["time_of_day"] = "night"

                        return row

                    steps_df = steps_df.apply(fix_time, axis=1)
                    hr_df = hr_df.apply(fix_time, axis=1)

                    # Convert the timestamp column to a datetime data type
                    steps_df['datetime'] = pd.to_datetime(steps_df['datetime'])
                    hr_df['datetime'] = pd.to_datetime(hr_df['datetime'])

                    # Drop any rows with missing or invalid data
                    steps_df.dropna(inplace=True)
                    hr_df.dropna(inplace=True)

                    # Merge the steps data and the heart rate data
                    merged_df = pd.merge(steps_df, hr_df, on='datetime')
                    merged_df = merged_df.drop(columns=['time_of_day_x', 'date_y', 'time_y'])

                    col = merged_df.pop('datetime')
                    merged_df.insert(0, 'datetime', col)

                    merged_df = merged_df.rename(columns={'date_x': 'date'})
                    merged_df = merged_df.rename(columns={'time_x': 'time'})
                    merged_df = merged_df.rename(columns={'time_of_day_y': 'time_of_day'})

                    # Calculate the average number of steps per day
                    meandaily_steps = merged_df.groupby(merged_df['datetime'].dt.date)['steps'].mean()

                    # Calculate the average heart rate for each day
                    meandaily_hr = merged_df.groupby(merged_df['datetime'].dt.date)['heartRate'].mean()

                    # Create a new DataFrame based on the groupby object
                    meandaily_steps_df = meandaily_steps.apply(lambda x: x).reset_index()
                    meandaily_hr_df = meandaily_hr.apply(lambda x: x).reset_index()

                    meandaily_steps_df = meandaily_steps_df.rename(columns={'datetime': 'date'})
                    meandaily_steps_df = meandaily_steps_df.rename(columns={'steps': 'meandailySteps'})

                    meandaily_hr_df = meandaily_hr_df.rename(columns={'datetime': 'date'})
                    meandaily_hr_df = meandaily_hr_df.rename(columns={'heartRate': 'meandailyHeartRate'})


                    # Plot the daily average steps data
                    chart1 = alt.Chart(meandaily_steps_df, title="Daily Average Steps").mark_line().encode(
                        x='date',
                        y='meandailySteps',
                        #color='Origin',
                    ).interactive()

                    tab11, tab12 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

                    with tab11:
                        # Use the Streamlit theme.
                        # This is the default. So you can also omit the theme argument.
                        st.altair_chart(chart1, theme="streamlit", use_container_width=True)
                    with tab12:
                        # Use the native Altair theme.
                        st.altair_chart(chart1, theme=None, use_container_width=True)


                    # Plot the daily average heart rate data
                    chart2 = alt.Chart(meandaily_hr_df, title="Daily Average Heart Rate (BPM)").mark_line().encode(
                        x='date',
                        y='meandailyHeartRate',
                        #color='Origin',
                    ).interactive()

                    tab21, tab22 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

                    with tab21:
                        # Use the Streamlit theme.
                        # This is the default. So you can also omit the theme argument.
                        st.altair_chart(chart2, theme="streamlit", use_container_width=True)
                    with tab22:
                        # Use the native Altair theme.
                        st.altair_chart(chart2, theme=None, use_container_width=True)


                    # Plot the steps data
                    chart3 = alt.Chart(merged_df, title="Steps Over Time").mark_line().encode(
                        x='datetime',
                        y='steps',
                        #color='Origin',
                    ).interactive()

                    tab31, tab32 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

                    with tab31:
                        # Use the Streamlit theme.
                        # This is the default. So you can also omit the theme argument.
                        st.altair_chart(chart3, theme="streamlit", use_container_width=True)
                    with tab32:
                        # Use the native Altair theme.
                        st.altair_chart(chart3, theme=None, use_container_width=True)


                    # Plot the heart rate data
                    chart4 = alt.Chart(merged_df, title="Heart Rate (BPM) Over Time").mark_line().encode(
                        x='datetime',
                        y='heartRate',
                        #color='Origin',
                    ).interactive()

                    tab41, tab42 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

                    with tab41:
                        # Use the Streamlit theme.
                        # This is the default. So you can also omit the theme argument.
                        st.altair_chart(chart4, theme="streamlit", use_container_width=True)
                    with tab42:
                        # Use the native Altair theme.
                        st.altair_chart(chart4, theme=None, use_container_width=True)


                    # Merge the average steps data and the heart rate data
                    mean_merged_df = pd.merge(meandaily_steps_df, meandaily_hr_df, on='date')

                    merged_df['date'] = pd.to_datetime(merged_df['date'], format='%Y-%m-%d')
                    df_date_counts = merged_df.groupby('date').size()

                    df_date_counts_df = df_date_counts.apply(lambda x: x).reset_index(name='count')

                    new_df = pd.DataFrame(columns=['date', 'meandailySteps', 'meandailyHeartRate'])
                    new_df['date'] = mean_merged_df['date'].astype(str).repeat(df_date_counts_df['count'])
                    new_df['meandailySteps'] = mean_merged_df['meandailySteps'].repeat(df_date_counts_df['count'])
                    new_df['meandailyHeartRate'] = mean_merged_df['meandailyHeartRate'].repeat(df_date_counts_df['count'])

                    # Convert the date column to datetime
                    new_df['date'] = pd.to_datetime(new_df['date'])

                    # Make indices unique
                    new_df = new_df.reset_index(drop=True)

                    result_df = pd.merge(merged_df, new_df, left_index=True, right_index=True)

                    result_df = result_df.drop(columns=['date_y'])
                    result_df = result_df.rename(columns={'date_x': 'date'})


                    # Convert the datetime values to the number of seconds since a reference point
                    result_df["datetime"] = pd.to_datetime(result_df["datetime"])
                    result_df["datetime"] = result_df["datetime"] - pd.to_datetime("1970-01-01")
                    result_df["datetime"] = np.round(result_df["datetime"].dt.total_seconds()).astype(int)


                    # Split the data into training and testing sets
                    train_result_df = result_df[:int(len(result_df) * 0.8)]
                    test_result_df = result_df[int(len(result_df) * 0.8):]

                    # Extract the datetime and steps columns as NumPy arrays
                    X_train = train_result_df[["datetime", "steps", "meandailySteps", "meandailyHeartRate"]].values
                    y_train = train_result_df["heartRate"].values

                    X_test = test_result_df[["datetime", "steps", "meandailySteps", "meandailyHeartRate"]].values
                    y_test = test_result_df["heartRate"].values

                    # Create the model
                    regressor = MLPRegressor(hidden_layer_sizes=(70,), max_iter=10, random_state=0)

                    # Train the model
                    regressor.fit(X_train, y_train)

                    # Predict the values for the test set
                    y_pred = regressor.predict(X_test)


                    # Calculate the mean squared error on the testing data
                    mse = np.mean((y_pred - y_test) ** 2)
                    st.write(f"Mean squared error: {mse:.2f}")

                    try:
                        # Prompt the user to enter a date and time
                        date_string = st.text_input("Enter a date and time (YYYY-MM-DD HH:MM:SS): ")

                        # Define the format of the date string
                        date_format = "%Y-%m-%d %H:%M:%S"

                        # Parse the date string and create a datetime object
                        date = datetime.datetime.strptime(date_string, date_format)

                        # Convert the datetime object to a Unix timestamp
                        timestamp = date.timestamp()

                        # (input datetime in epoch)
                        input_epoch_timecode = timestamp
                        # (input steps)
                        input_steps = st.number_input("Enter steps: ")
                        # (input daily steps)
                        input_meandaily_steps = st.number_input("Enter average steps: ")
                        # (input daily heart rate)
                        input_meandaily_hr = st.number_input("Enter average heart rate: ")

                        # Convert an epoch Unix timecode to a datetime object
                        specific_date = pd.to_datetime(input_epoch_timecode, unit='s')

                        # Convert the specific date to a numeric representation
                        specific_date_seconds = np.round((specific_date - pd.to_datetime("1970-01-01")).total_seconds()).astype(int)


                        # Use the model to predict the value for the specific date
                        prediction = regressor.predict(np.array([[specific_date_seconds, input_steps, input_meandaily_steps, input_meandaily_hr]]))[0]

                        # Print the prediction
                        first_row_user_df = user_df.iloc[0]
                        st.write(f"User ID of: {first_row_user_df.userId} - ({first_row_user_df.nickName})")
                        st.write(f"Heart rate prediction for {specific_date}: {prediction:.2f}")

                    except:
                        st.error("Please enter the correct date and time format.")

                except:
                    st.error("Something went wrong.")
            
            elif submitted and not path1 and not path2 and not path3:
                st.error("File not found.")



def process_file(file):
    body = st.empty()

    with body.container():
        st.title(f"""Please wait... your file is being processed...""")



with st.container():
    upload_file()

import os
from flask import Flask, send_file, render_template
import pandas as pd
import numpy as np
import utils
from io import BytesIO


app = Flask(__name__)
app.secret_key = 'super secret'

YEARS = [2015, 2016, 2017, 2018, 2019, 2020]
TARGET_YEAR = 2022


@app.route('/')
def index():
    global YEARS, TARGET_YEAR

    return render_template('index.html', train_years = ', '.join(str(y) for y in YEARS), target_year = TARGET_YEAR)

@app.route("/generate/",methods=['GET'])
def generate_report():
    global YEARS, TARGET_YEAR
    
    results = utils.predict(YEARS)
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    results.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    output.seek(0)
    return send_file(output, attachment_filename=f'{TARGET_YEAR}_report.xlsx', as_attachment=True)


if __name__ == "__main__":
    app.run()

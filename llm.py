import google.generativeai as genai
import os
import pandas as pd
import textwrap
import re
from IPython.display import display, Markdown
import os

os.environ['GOOGLE_API_KEY'] = 'AIzaSyD_nxEn0-nWtnp9TK1V5z6GQlE8-hyZRIo'

generation_config={
"temperature":0,

}


genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel(model_name='gemini-pro',generation_config=generation_config)


def analyze_data(df):


        query='''I want you to act like a doctor. I will give you summary of a patient's stay in the hospital, you will evaluate it and answer a set of questions as yes or no unless any other format is mentioned along with the question.
        1.Is there any mention of consultation by nephrologist for this patient.
        2.If yes then when was it done, answer as a timestamp.
        3.Is it explicitly mentioned to avoid nephrotoxic drugs for the patient. 
        4.Is it mentioned that the patient has Acute Kidney Injury (AKI).
        5.Was the patient put under General Anaesthesia at any point.
        6.Has hypertension been mentioned as a previously existing condition in the patient.
        7. Is patient's Potassium level consistently low
        8. Is patient's Potassium level consistently high
        9. Is patient's Sodium level consistently low
        10. Is patient's Sodium level consistently high
        11. Has the patient been advised to reduce fluid intake
        12.Has this patient undergone angiography.
        13.Is the patient being given any diuretic
        14.Has the patient undergone been injected with any kind of contrast dye
        15.As per this summary was the patient ever admitted to the ICU
        16.Is there any mention of patient having Chronic Obstructive Pulmonary disease (COPD)
        17.Was the patient put on Ventilator during his/her stay in the hospital 
        18.Did the patient develop Tachycardia at any point during his/her stay in the hospital
        19.Is there any mention of drop in Oxygen saturation, 
        20.Is there any mention of patient developing Cardio-Renal syndrome.
        Only return the answers, next to indices number and not the questions
        '''


        # Assuming 'df' is the DataFrame containing the patient summaries
        # Initialize an empty DataFrame to store the results
        results_df = pd.DataFrame(columns=[
        "Consultation by Nephrologist",
        "Timestamp of Consultation",
        "Avoid Nephrotoxic Drugs",
        "AKI Mentioned",
        "General Anaesthesia",
        "Hypertension Mentioned",
        "Hyperkalemia Mentioned",
        "Hypokalemia Mentioned",
        "Hypernatremia Mentioned",
        "Hyponatremia Mentioned",
        "Fluid Restriction Advised",
        "Angiography Done",
        "Diuretic Given",
        "Imaging Procedure with Contrast",
        "ICU Admission",
        "COPD Mentioned",
        "Ventilator Used",
        "Tachycardia Developed",
        "Oxygen Saturation Drop Mentioned",
        "Cardio-Renal Syndrome Mentioned"
        ])
        c=0

        # Iterate through each entry in the 'Course In Hospital' column
        for cih in df['Course In Hospital']:
            if c==10:
                break
            if cih:
                text = cih
                response = model.generate_content([query, text])
                response.resolve()
                response_list = [value.split('.', 1)[1].strip() for value in response.text.split('\n') if value]

                # Extract the answers from the response_list
                # Start from index 1 as index 0 is the question
                answers = response_list
            else:
                answers=['NA']*20

            # Assuming 'answers' is a list containing the responses in order
            result_row = {
                "Patient_id":df['Patient ID'][c],
                "Consultation by Nephrologist": answers[0],
                "Timestamp of Consultation": answers[1],
                "Avoid Nephrotoxic Drugs": answers[2],
                "AKI Mentioned": answers[3],
                "General Anaesthesia": answers[4],
                "Hypertension Mentioned": answers[5],
                "Hyperkalemia Mentioned": answers[6],
                "Hypokalemia Mentioned": answers[7],
                "Hypernatremia Mentioned": answers[8],
                "Hyponatremia Mentioned": answers[9],
                "Fluid Restriction Advised": answers[10],
                "Angiography Done": answers[11],
                "Diuretic Given": answers[12],
                "Imaging Procedure with Contrast": answers[13],
                "ICU Admission": answers[14],
                "COPD Mentioned": answers[15],
                "Ventilator Used": answers[16],
                "Tachycardia Developed": answers[17],
                "Oxygen Saturation Drop Mentioned": answers[18],
                "Cardio-Renal Syndrome Mentioned": answers[19]
                
            }

            # Concatenate the results for this entry to the DataFrame
            results_df = pd.concat([results_df, pd.DataFrame(result_row, index=[0])], ignore_index=True)
            c+=1
            print('pdf',c)
        return results_df


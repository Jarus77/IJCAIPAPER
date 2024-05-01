import streamlit as st
import os
import pandas as pd
import pdfplumber
import re
import tempfile



def process_pdf(file_path):

        # Define the list of keywords to stop extraction
        stop_keywords = [
        "Other Consultants Attended the Case",
        "Discharge Date",
        "Diagnosis",
        "Chief Complaints",
        "Past History",
        "Significant Findings",
        "Course In Hospital",
        "Investigations",
        "Treatment/ Drugs Given during Stay",
        "Patient Response",

        "Medication On Discharge",
        "Surgery / Procedure",
        "Treatment/ Drugs Given during Stay",
        "Patient Response",
        "Medication On Discharge",
        "Instructions To Patient",
        "Urgent Care Advice",
        "Signature",
        ]

        # Path to the folder containing PDFs


        # Initialize an empty DataFrame outside the loop
        df = pd.DataFrame(columns=[
        "Course In Hospital",
        "Diagnosis",
        "Chief Complaints",
        "Surgery / Procedure",
        "Status On Discharge",
        "Past History",
        "Patient ID",
        "DOB",
        "Gender",
        "Age",
        "Speciality",
        "admission_date",
        "discharge_date"
        ])

        # Iterate through each file in the folder

                # Construct the full path to the PDF file
        pdf_file_path = file_path

        # Initialize dictionaries to store the extracted data for this PDF
        data_dict = {
            "Course In Hospital": [],
            "Diagnosis": [],
            "Chief Complaints": [],
            "Surgery / Procedure": [],
            "Status On Discharge": [],
            "Past History": [],
            "Patient ID": [],
            "DOB": [],
            "Gender": [],
            "Age": [],
            "Speciality": [],
            "admission_date": [],
            "discharge_date": []
        }

        try:
            # Perform the operations for each PDF file
            with pdfplumber.open(pdf_file_path) as pdf:
                extracted_text = ""

                # Iterate through each page and extract text
                for page in pdf.pages:
                    page_text = page.extract_text()
                    extracted_text += page_text

            # Replace newline characters with a space in the extracted text
            text = extracted_text.replace('\n', ' ')

            # ... (rest of your code for extracting information)
            
            # Create a regex pattern that matches any of the stop keywords
            stop_pattern = '|'.join(map(re.escape, stop_keywords))
            # Define regular expressions to extract information
            patient_id_match = re.search(r'Patient ID : (.+?)\s', text)
            dob_match = re.search(r'Date of Birth : (.+?)\s', text)
            gender_match = re.search(r'Gender : (.+?)\s', text)
            age_match = re.search(r'Age : (\d+)', text)
            speciality_match = re.search(r'Specialty : (.+?)\s', text)
            admission_date_match = re.search(r'Admission : (\d{2}/\d{2}/\d{4} \d{2}:\d{2})', text)
            discharge_date_match = re.search(r'Discharge Date : (\d{2}/\d{2}/\d{4})', text)

            # Check if the matches are not None before accessing the group attribute
            if patient_id_match:
                data_dict["Patient ID"].append(patient_id_match.group(1))
            else:
                data_dict["Patient ID"].append(None)

            if dob_match:
                data_dict['DOB'].append(dob_match.group(1))
            else:
                data_dict['DOB'].append(None)

            if gender_match:
                data_dict['Gender'].append(gender_match.group(1))
            else:
                data_dict['Gender'].append(None)

            if age_match:
                data_dict['Age'].append(age_match.group(1))
            else:
                data_dict['Age'].append(None)

            if speciality_match:
                data_dict['Speciality'].append(speciality_match.group(1))
            else:
                data_dict['Speciality'].append(None)

            if admission_date_match:
                data_dict['admission_date'].append(admission_date_match.group(1))
            else:
                data_dict['admission_date'].append(None)

            if discharge_date_match:
                data_dict['discharge_date'].append(discharge_date_match.group(1))
            else:
                data_dict['discharge_date'].append(None)

            # Extract the text sections and store them in the respective dictionaries
            course_in_hospital_match = re.search(r'Course In Hospital : ([\s\S]*?)(?=' + stop_pattern + r'|\Z)', text)
            if course_in_hospital_match:
                data_dict["Course In Hospital"].append(course_in_hospital_match.group(1).strip())
            else:
                data_dict["Course In Hospital"].append(None)

            diagnosis_match = re.search(r'Diagnosis : ([\s\S]*?)(?=' + stop_pattern + r'|\Z)', text)
            if diagnosis_match:
                data_dict["Diagnosis"].append(diagnosis_match.group(1).strip())
            else:
                data_dict["Diagnosis"].append(None)

            chief_complaints_match = re.search(r'Chief Complaints : ([\s\S]*?)(?=' + stop_pattern + r'|\Z)', text)
            if chief_complaints_match:
                data_dict["Chief Complaints"].append(chief_complaints_match.group(1).strip())
            else:
                data_dict["Chief Complaints"].append(None)

            surgery_match = re.search(r'Surgery \/ Procedure : ([\s\S]*?)(?=' + stop_pattern + r'|' + re.escape("Status On Discharge:") + r'|\Z)', text)
            if surgery_match:
                data_dict["Surgery / Procedure"].append(surgery_match.group(1).strip())
            else:
                data_dict["Surgery / Procedure"].append(None)

            status_on_discharge_match = re.search(r'Status On Discharge : ([\s\S]*?)(?=' + stop_pattern + r'|\Z)', text)
            if status_on_discharge_match:
                data_dict["Status On Discharge"].append(status_on_discharge_match.group(1).strip())
            else:
                data_dict["Status On Discharge"].append(None)

            past_history_match = re.search(r'Past History : ([\s\S]*?)(?=' + stop_pattern + r'|\Z)', text)
            if past_history_match:
                data_dict["Past History"].append(past_history_match.group(1).strip())
            else:
                data_dict["Past History"].append(None)

            max_length = max(len(lst) for lst in data_dict.values())

            # Fill missing data with None to make all lists of the same length
            for key in data_dict:
                if len(data_dict[key]) < max_length:
                    data_dict[key] += [None] * (max_length - len(data_dict[key]))

            # Append the data for this PDF to the DataFrame
            df = pd.concat([df, pd.DataFrame(data_dict)], ignore_index=True)

        except Exception as e:
            print(f"Error processing")

        return df

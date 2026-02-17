import csv; [print(float(row['avg_response_time_ms'])/int(row['batch_size'])) for row in csv.DictReader(open('results.csv'))]

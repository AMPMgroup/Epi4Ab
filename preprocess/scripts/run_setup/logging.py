import os
from datetime import date

class DataLogging():
    def __init__(self, args):
        self.directory_metadata = args.directory_metadata
        self.directory_data = args.directory_output

        self.download_pdb = args.download_pdb

        if not os.path.exists(self.directory_data):
            os.mkdir(self.directory_data)

        self.run_date = date.today()

        # error
        self.error_download = []
        self.error_extract_structure = []
        self.error_extract_cdrs = []
        self.error_pdb2pqr = []
        self.error_sasa = []
        self.error_depth = []
        self.error_angle = []
        self.error_charge = []
        self.error_charge_compostion = []
        self.error_interface = []
        self.error_ellipro = []
        self.error_gather = []
        self.error_empty_interface = []
        self.error_sphere = []

        self.total_time = None
        self.message = f'''
Run date: {self.run_date}'''

    def save_log(self):
        self.message += f'''
Total runtime: {self.total_time}'''
        print(self.message)
        # with open(os.path.join(self.directory_output, 'error.txt'), 'a') as f:
        #     f.write(self.message)

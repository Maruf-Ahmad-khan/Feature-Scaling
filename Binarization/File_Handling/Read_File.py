class File_Handlers():
     
     def __init__(self, File_Name):
          self.File_Name = File_Name
          
          
     def Read_File(self):
          """_summary_
               Write the code to read the 
               File contents
          """
          with open(self.File_Name, 'r') as file:
               return file.read()
          
          
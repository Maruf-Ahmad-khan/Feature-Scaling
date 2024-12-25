class File_Handler():
     
     def __init__(self, file_name):
          self.file_name = file_name
          
     def Write_to_File(self, content):
          
          """
          Create a sample.txt file and write something in it.
     
          """
          with open(self.file_name, 'w') as file:
               file.write(content)
               
               

     
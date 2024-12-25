from Write_to_file import File_Handler

# Driver code

if __name__ == "__main__":
     file = File_Handler("Sample.txt")
     ans = file.Write_to_File("\nHello I am learning python from the scratch.")
     print(ans)
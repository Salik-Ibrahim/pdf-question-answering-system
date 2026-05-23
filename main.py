import pymupdf

doc = pymupdf.open("Documents/Test.pdf")  # opening a document(pdf)
print(doc)
print(len(doc))
str = ""
for i in range(len(doc)):
    str += doc[i].get_text()
print(str[:500])

'''Napišite program koji od korisnika zahtijeva unos brojeva u beskonacnoj petlji ˇ
sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
(npr. slovo umjesto brojke) na nacin da program zanemari taj unos i ispiše odgovaraju ˇ cu poruku.'''

numbers = []

while True:
    inputData = input("Unesi broj ili 'Done'.")
    if inputData == "Done":
        break

    try:
        number = float(inputData)
        numbers.append(number)
    except ValueError:
        print("Nije unesen broj!")

print(f"Length of numbers list: {len(numbers)}")
print(f"Average: {sum(numbers) / len(numbers)}")
print(f"Min: {min(numbers)}")
print(f"Max: {max(numbers)}")
numbers.sort()
print(f"Sorted list: {numbers}")

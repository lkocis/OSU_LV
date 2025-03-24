'''Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je placen ´
po radnom satu. Koristite ugradenu Python metodu ¯ input(). Nakon toga izracunajte koliko ˇ
je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na nacin da ukupni iznos ˇ
izracunavate u zasebnoj funkciji naziva ˇ total_euro.
Primjer:
Radni sati: 35 h
eura/h: 8.520
Ukupno: 297.5 eura'''

work_hours = float(input("Unesi broj radnih sati: "))
pay_per_hour = float(input("Unesi cijenu po satu: "))

def total_euro(work_hours, pay_per_hour):
    return work_hours * pay_per_hour

total = total_euro(work_hours, pay_per_hour)
print(f"Total pay is: {total}")

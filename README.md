# Wheelczur

Program do śledzenia kończyn psa na nagraniach wideo i uczenia klasyfikatorów chodu `zdrowy/chory`. Wykorzystuje modele YOLOX-Tiny i RTMPose, a wyniki zapisuje jako dane CSV oraz film z zaznaczonymi punktami.

## 1. Instalacja Pythona

Program działa na Windowsie i Linuxie, wymaga Pythona **3.10–3.13** oraz internetu podczas instalacji.

### Windows

1. Na [python.org](https://www.python.org/downloads/windows/) znajdź **Python 3.13.x** i pobierz **Windows installer (64-bit)**.
2. Uruchom instalator, zaznacz **Add python.exe to PATH** i kliknij **Install Now**. Narzędzie `pip` zainstaluje się razem z Pythonem.
3. Otwórz nowe okno PowerShell i sprawdź instalację:

```powershell
python --version
python -m pip --version
```

Pierwsze polecenie powinno wyświetlić wersję `3.13.x`.

### Linux (Ubuntu 24.04 LTS / Debian 13)

Otwórz terminal i zainstaluj Pythona, `pip` oraz obsługę środowisk wirtualnych:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

Wersja musi mieścić się w zakresie **3.10–3.13**. W dalszych poleceniach na Linuxie używaj **`python3` zamiast `python`**. Na innych dystrybucjach zainstaluj zgodną wersję Pythona i moduł `venv` przez menedżer pakietów systemu.

### Brak polecenia `pip` lub modułu `pip`

Jeśli terminal zgłasza, że `pip` nie jest rozpoznawany, najpierw sprawdź `python -m pip --version` (na Linuxie: `python3 -m pip --version`). Jeśli to działa, używaj `python -m pip` zamiast samego `pip`.

**Windows — komunikat `No module named pip`:** zamknij menu launchera opcją **0**, a następnie wykonaj w PowerShell:

```powershell
python -m ensurepip --upgrade
python -m pip --version
```

Jeśli polecenie `python` nie działa, ale działa `py`, zastąp `python` przez `py` we wszystkich poleceniach. Jeśli brakuje również modułu `ensurepip`, uruchom ponownie instalator tej samej wersji Pythona 3.10–3.13 z [python.org](https://www.python.org/downloads/windows/), wybierz **Modify** i zaznacz składnik **pip**. Po instalacji otwórz nowe okno terminala i ponownie sprawdź wersję `pip`.

**Linux (Ubuntu / Debian) — brak `pip` lub `ensurepip`:** zainstaluj wymagane pakiety przez menedżer systemowy:

```bash
sudo apt update
sudo apt install python3-pip python3-venv
python3 -m pip --version
```

Jeśli korzystasz z innej wersji interpretera, np. `python3.12`, doinstaluj odpowiadający jej pakiet `python3.12-venv`.

**Brak `pip` w istniejącym środowisku Wheelczura:** jeśli błąd pochodzi z `.venv-cpu`, przywróć `pip` bezpośrednio w tym środowisku:

Windows (PowerShell):

```powershell
.\.venv-cpu\Scripts\python.exe -m ensurepip --upgrade
.\.venv-cpu\Scripts\python.exe -m pip --version
```

Linux:

```bash
./.venv-cpu/bin/python -m ensurepip --upgrade
./.venv-cpu/bin/python -m pip --version
```

Dla GPU zamień `.venv-cpu` na `.venv-cuda`. Jeśli brakuje `ensurepip`, najpierw napraw instalację Pythona zgodnie z instrukcją dla swojego systemu powyżej. Następnie ponów `python launcher.py --setup cpu` (dla GPU: `--setup cuda`; na Linuxie użyj `python3`).

Opis mechanizmu przywracania `pip`: [dokumentacja Pythona — ensurepip](https://docs.python.org/3.13/library/ensurepip.html).

## 2. Instalacja programu

Na GitHubie wybierz **Code → Download ZIP** i rozpakuj archiwum. Otwórz terminal w folderze zawierającym `launcher.py` i wykonaj kolejno:

```bash
python launcher.py --setup cpu
python launcher.py --models
```

Pierwsze polecenie tworzy środowisko `.venv-cpu` i instaluje wszystkie wymagane pakiety, w tym `numpy` i `opencv-python-headless`. Drugie pobiera i sprawdza modele. Nie trzeba ręcznie aktywować środowiska.

### Błąd certyfikatu przy pobieraniu modeli (Windows)

Jeśli przy pobieraniu modeli pojawi się `CERTIFICATE_VERIFY_FAILED` lub `unable to get local issuer certificate`, spróbuj poniższego rozwiązania. Wyjdź z menu launchera opcją **0**, jeśli jest otwarte. Następnie w terminalu, w folderze projektu, wykonaj kolejno:

```bat
python -m pip install --upgrade truststore
python -c "import truststore; truststore.inject_into_ssl(); import download_models; raise SystemExit(download_models.main())"
```

**Wykonaj obie komendy** — druga uruchamia pobieranie z użyciem certyfikatów systemowych. Samo zainstalowanie `truststore` nie zmienia działania opcji 7 w launcherze.

Po komunikacie **Oba modele są gotowe** uruchom ponownie `python launcher.py` i kontynuuj pracę z menu.

## 3. Użycie

```bash
python launcher.py --run cpu
```

1. Podaj ścieżkę do filmu, np. `test-vid/doggi-walk-01.mp4`.
2. Wybierz jedną z czterech kończyn: `left-fore`, `right-fore`, `left-hind` lub `right-hind`. Lewa i prawa strona odnoszą się do stron psa.
3. Poczekaj na zakończenie analizy.

Wyniki znajdziesz obok filmu, w folderze `<nazwa_filmu_bez_rozszerzenia>_tracking/`:

- **CSV** — współrzędne punktów, czas i pewność wykrycia.
- **MP4** — film z zaznaczonymi punktami i połączeniami.
- **JSON** — ustawienia i podsumowanie analizy.

### Obsługa przez menu tekstowe

Po zainstalowaniu Pythona możesz wykonać wszystkie kroki z menu tekstowego. W folderze projektu wystarczy uruchomić:

```bash
python launcher.py
```

Menu umożliwia instalację pakietów, pobranie modeli, analizę na CPU/GPU, trening oraz sprawdzenie środowiska. Wpisz numer wybranej opcji i zatwierdź klawiszem **Enter**. Przy pierwszym użyciu na CPU wybierz kolejno **1 → 7 → 3**. Do treningu służą opcje **9 (CPU)** i **10 (GPU)**.

## 4. Trening modelu na nagraniach zdrowych i chorych psów

Po aktualizacji projektu ponów instalację środowiska, aby dodać pakiet `scikit-learn`:

```bash
python launcher.py --setup cpu
python launcher.py --models
python launcher.py
```

Dla NVIDIA użyj `--setup cuda`. W menu wybierz **9** albo **10**, a następnie podaj ścieżkę do folderu z nagraniami. Ścieżkę ze spacjami możesz wkleić w cudzysłowie. Wskazujesz folder nadrzędny, zawierający **oba** poniższe podfoldery:

| Ścieżka w folderze treningowym | Zawartość |
| --- | --- |
| `zdrowy/` | Czyste nagrania `.mp4` zdrowych psów |
| `chory/` | Czyste nagrania `.mp4` chorych psów |
| `zdrowy/reksio/*.mp4` | Opcjonalnie: kilka nagrań jednego psa w jego podfolderze |
| `chory/azor/*.mp4` | Opcjonalnie: nagrania innego psa w osobnym podfolderze |

Obsługiwane są również rozszerzenia `.MP4` i podfoldery. Dla wiarygodnej walidacji grupuj nagrania według psów: pierwszy podfolder w `zdrowy/` lub `chory/` identyfikuje psa. Używaj tej samej nazwy dla tego samego psa w obu klasach i różnych nazw dla różnych psów. Bez tych podfolderów każde nagranie jest osobną grupą, więc program nie rozpozna, że dwa filmy przedstawiają tego samego psa.

### Co robi trening

1. Wykrywa i śledzi psa oraz określa kierunek jego ustawienia na podstawie osi tułowia. Początkowym celem jest największy wykryty pies. Używaj nagrań jednego psa idącego z boku, w jednym kierunku; przejścia z zawracaniem podziel na osobne filmy.
2. Przeskalowuje i centruje każdą klatkę na obrazie **640 × 640**, zachowując proporcje. Dłuższy bok obszaru psa ma **480 px**. Przejścia w lewo są odbijane poziomo, więc pies jest ustawiony w prawo.
3. Na znormalizowanych klatkach RTMPose śledzi jednocześnie przednią i tylną kończynę od strony kamery. Dla psa ustawionego w prawo wybierana jest anatomiczna prawa strona, a dla psa ustawionego w lewo — lewa, która po odbiciu ma indeksy prawej strony obrazu kanonicznego. Wybrana strona jest stała dla nagrania. Kończyny z tła nie zastępują punktów niewidocznej kończyny bliskiej.
4. Zapisuje osobno `front.csv` i `hind.csv`. Kolumna **`which_limb`** zawiera odpowiednio **`front`** lub **`hind`**. `limb` opisuje kończynę po normalizacji (`right-fore`/`right-hind`), a `source_limb` jej stronę w oryginalnym nagraniu. Zwykły tracker również eksportuje teraz `which_limb`.
5. Czyta zapisane CSV i wylicza cechy ruchu: położenie łapy względem barku/biodra, kąty, wyprost, prędkości i okresowość. Analizuje okna do 2 sekund z krokiem 1 sekundy, po przeliczeniu osi czasu do 30 Hz. Nie wykorzystuje nazwy filmu, etykiety ani tła jako cech wejściowych modelu.
6. Uczy **dwa osobne klasyfikatory Random Forest**, dla `front` i `hind`, z etykietami `zdrowy/chory`. Zapisuje je automatycznie w **`model/model_V001/`**, następnie **`model/model_V002/`** itd. Numer jest większy od dotychczasowych wersji; poprzednie modele są zachowywane. Niepełny trening nie tworzy gotowej wersji.

Automatyczny wybór bliskiej strony opiera się na założeniu ujęcia z boku, nie na pomiarze głębi. Przy ujęciu skośnym, zasłonięciu łap albo błędzie RTMPose nie można zagwarantować prawidłowego rozdzielenia kończyn. **Sprawdź filmy `tracking.mp4` zapisane przy CSV.** Niepewne punkty mają puste współrzędne. Klatki bez psa lub z niepewnym kierunkiem są pomijane w cechach; w znormalizowanym filmie zachowują swój czas jako czarne klatki. Jeżeli kierunek wyraźnie zmienia się w nagraniu, film zostaje odrzucony.

Nagranie musi dostarczyć co najmniej 20 kompletnych klatek każdej kończyny i co najmniej 50% kompletnych klatek. Okna cech nie łączą luk w śledzeniu dłuższych niż 150 ms; wymagają też przynajmniej 0,5 sekundy ruchu. Odrzucone nagrania i przyczyny zapisują się w `dataset_manifest.json`. Jeśli zabraknie którejkolwiek klasy, trening kończy się błędem bez zapisu modelu.

Etykieta folderu dotyczy całego nagrania i jest używana dla obu klasyfikatorów. Sam podział `zdrowy/chory` nie wskazuje, która konkretna kończyna jest chora. Trening uczy rozpoznawania klasy chodu na podstawie ruchu wybranej kończyny; modele ONNX wykrywające psa i punkty pozostają gotowymi trackerami.

### Walidacja i wyniki

Gdy są co najmniej cztery niezależne grupy i co najmniej dwie grupy każdej klasy, program próbuje odłożyć około 25% grup do walidacji, zachowując obie klasy po obu stronach podziału. Wszystkie okna, nagrania i obie kończyny tej samej grupy pozostają po jednej stronie. Wynik liczony jest na poziomie nagrania; długie nagranie nie daje osobnego głosu każdej klatce. Jeśli bez rozdzielania grup nie da się wydzielić obu klas, trening działa, ale **nie podaje skuteczności walidacyjnej**. Po walidacji końcowe modele uczą się na wszystkich przyjętych nagraniach.

| Plik w `model/model_Vxxx/` | Zawartość |
| --- | --- |
| `front_model.joblib` | Wyuczony klasyfikator, `which_limb=front`, kolejność cech i ustawienia |
| `hind_model.joblib` | Wyuczony klasyfikator, `which_limb=hind`, kolejność cech i ustawienia |
| `metadata.json` | Ustawienia, wersje bibliotek, podział grup i wyniki walidacji |
| `training_features.csv` | Cechy uczące z `which_limb`, etykietami i przypisaniem do walidacji |
| `validation_predictions.csv` | Przewidywania na odłożonych nagraniach, jeśli walidacja była możliwa |
| `dataset_manifest.json` | Lista przyjętych i odrzuconych nagrań |
| `data/<klasa>/<identyfikator>/` | `normalized.mp4`, `tracking.mp4`, `front.csv`, `hind.csv`, `normalization.json` |

Pliki CSV zawierają współrzędne w **znormalizowanym** obrazie. `normalization.json` zapisuje FPS, odbicie, źródłową stronę psa, rozmiar oraz liczbę poprawnych klatek. Nagrania wynikowe nie zawierają dźwięku.

### Trening bez menu

```bash
python launcher.py --train cpu --dataset "D:/nagrania/trening"
python launcher.py --train cuda --dataset "D:/nagrania/trening"
```

Opcja `--model-output-root "D:/wyniki/model"` zmienia folder na wersje modeli. Domyślnie `model/` znajduje się obok `launcher.py`, niezależnie od folderu nagrań.

Po aktywacji zainstalowanego środowiska dostępne jest również polecenie z parametrami okien i progów:

```bash
python train_gait.py "D:/nagrania/trening" --device cpu --window-seconds 2 --stride-seconds 1
```

Zapisany model można ponownie wczytać i użyć do oceny CSV w tym samym formacie normalizacji:

```bash
python train_gait.py --predict-csv "model/model_V001/data/zdrowy/IDENTYFIKATOR/front.csv" --which-limb front --model-version model/model_V001
```

Zastąp `IDENTYFIKATOR` nazwą podfolderu z wynikami. Analogicznie dla tylnej kończyny użyj `hind.csv` i `--which-limb hind`. Polecenie zwraca klasę oraz uśrednione wyniki klasyfikatora. Zwykły CSV trackera w oryginalnych pikselach wymaga najpierw normalizacji i nie jest przyjmowany jako CSV treningowy.

Dokumentacja użytych formatów i narzędzi: [punkty AP-10K w MMPose](https://mmpose.readthedocs.io/en/latest/_modules/mmpose/datasets/datasets/animal/ap10k_dataset.html), [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).

## 5. Ocena nowego nagrania MP4 wytrenowanymi modelami

Dotychczas zapisane `front_model.joblib` i `hind_model.joblib` są obsługiwane bez ponownego treningu. W `model/model_Vxxx/` muszą znajdować się oba te pliki oraz `metadata.json` z tego samego treningu.

1. Uruchom `python launcher.py`.
2. Wybierz **11 — ocena na CPU** albo **12 — ocena z trackerem CUDA**.
3. Wybierz wytrenowaną wersję z listy. **Enter wybiera najnowszą kompletną wersję**. Możesz też wpisać nazwę `model_V001` albo ścieżkę do wersji zapisanej w innym folderze.
4. Wklej ścieżkę do nowego `.mp4`, np. `D:/nagrania/test/pies.mp4`. Nagranie do oceny może znajdować się w dowolnym folderze; nie wymaga podziału na `zdrowy/` i `chory/`.
5. Poczekaj na normalizację i tracking. Program odczyta cechy i wyświetli wyniki `zdrowy` oraz `chory` w procentach, osobno dla `front` i `hind`.

**Użyj filmu, którego model nie widział podczas treningu**, jeżeli chcesz sprawdzić działanie na nowych danych. Do oceny nie są potrzebne oryginalne nagrania treningowe ani wygenerowane z nich CSV.

Ocena korzysta z tej samej normalizacji, progów trackera i ustawień cech co zapisana wersja. Wczytuje klasyfikatory; nie uruchamia ich uczenia i nie zapisuje niczego w wybranej wersji modelu. Gdy materiał nie dostarcza dostatecznej liczby wiarygodnych punktów obu bliskich kończyn, zgłasza przyczynę zamiast podawać procenty.

### Jak czytać procenty

Każdy klasyfikator zwraca dwa wyniki, które sumują się do 100%. Program uśrednia je po poprawnych oknach ruchu w filmie i wskazuje klasę z wyższym wynikiem. Dokładny remis 50/50 jest oznaczony jako niejednoznaczny. Osobno pokazuje odpowiedź przedniej i tylnej kończyny oraz informację, czy modele są zgodne. Nie łączy ich w jedno wymyślone prawdopodobieństwo choroby psa.

**Procenty to wyniki klasyfikatorów, nie potwierdzone prawdopodobieństwo choroby.** Wysoki wynik na filmie treningowym nie dowodzi dobrej skuteczności na nowych psach. Raport pokazuje osobno wynik walidacji zapisanej wersji, jeśli taki wynik był dostępny. Pamiętaj też, że foldery treningowe oznaczały stan całego nagrania, a nie diagnozę konkretnej kończyny.

### Gdzie są wyniki oceny

Obok ocenianego filmu powstaje folder `<nazwa_filmu>_ocena/`. Każda ocena otrzymuje własny podfolder `ocena_<data>_<identyfikator>/`, więc można powtarzać sprawdzanie różnych wersji bez nadpisywania raportów.

| Plik w folderze jednej oceny | Zawartość |
| --- | --- |
| `raport.html` | Czytelny raport z procentami obu klas, informacją o walidacji i podglądem trackingu; otwórz dwuklikiem |
| `wyniki.csv` | Dwa wiersze: `front` i `hind`, wskazana klasa oraz procenty `zdrowy/chory` |
| `wyniki.json` | Pełne wyniki, ustawienia, jakość trackingu, wersja oraz sumy SHA-256 użytych klasyfikatorów |
| `wyniki_okien.csv` | Wyniki klas dla kolejnych okien ruchu; wartości od 0 do 1 |
| `tracking/` | Znormalizowany film, `tracking.mp4` z punktami, `front.csv`, `hind.csv`, `normalization.json` |

Niektóre przeglądarki nie odtwarzają kodeka MP4 używanego przez OpenCV. Wtedy otwórz `tracking/tracking.mp4` w odtwarzaczu, np. VLC; procenty i pozostała część raportu HTML nadal są czytelne.

### Ocena bez menu

Zwykły launcher dobierze wcześniej zainstalowane środowisko:

```bash
python launcher.py --evaluate cpu --video "D:/nagrania/test/pies.mp4" --model-version model/model_V001
```

Zastąp `cpu` przez `cuda`, aby tracker użył NVIDIA GPU 0. Parametr `--evaluation-output-dir "D:/wyniki/oceny"` zmienia nadrzędny folder raportów. Gdy pominięto `--model-version` lub `--video`, launcher o nie zapyta.

Bezpośrednio w środowisku, w którym wykonywano trening, można użyć:

```bash
python evaluate_gait.py "D:/nagrania/test/pies.mp4" --model-version model/model_V001 --device cpu
```

W tym poleceniu pominięcie `--model-version` oznacza automatyczny wybór najnowszej kompletnej wersji w `model/`. Jeżeli zapisany klasyfikator pochodzi z innej wersji `scikit-learn` niż środowisko oceny, program podaje numer pakietu potrzebny do wczytania; trening nie musi być ponawiany. Najprościej zachować środowisko, w którym dany model powstał.

### Aktualizacja projektu i zachowanie wytrenowanych modeli

Najprościej rozpakuj paczkę aktualizacyjną **bezpośrednio do obecnego folderu z `launcher.py`**, zgadzając się na zastąpienie plików kodu. Paczka aktualizacyjna nie zawiera folderu `model/`, środowisk ani wag ONNX. Nie wymaga nowych zależności względem poprzedniej wersji z treningiem.

Jeśli tworzysz osobny folder z nową kopią projektu, przenieś dane według tabeli:

| Z dotychczasowego projektu | Do nowej kopii |
| --- | --- |
| `model/` | `model/` obok nowego `launcher.py`; zachowaj nazwy `model_Vxxx` |
| `canine_limb/models/rtmpose_ap10k.onnx` | Ten sam plik w `canine_limb/models/` |
| `canine_limb/models/yolox_tiny.onnx` | Ten sam plik w `canine_limb/models/` |

Minimalnie jedna wersja wymaga wspólnie `front_model.joblib`, `hind_model.joblib` i `metadata.json`. Kopia całego `model/` zachowuje także dane i historię treningu. Środowisko wirtualne w nowej lokalizacji przygotuj launcherem (`--setup cpu` albo `--setup cuda`); nie przenoś samego venv między ścieżkami. Jeśli instalowana wersja `scikit-learn` różni się od użytej w treningu, dopasuj ją do pola `versions.sklearn` w `metadata.json`.

## Opcjonalnie: karta NVIDIA (CUDA)

Po pobraniu modeli zamiast poleceń CPU użyj:

```bash
python launcher.py --setup cuda
python launcher.py --run cuda
```

Instalator automatycznie doda `onnxruntime-gpu` z bibliotekami CUDA/cuDNN. Wymagana jest karta NVIDIA z odpowiednim sterownikiem, a w Windows także Visual C++ x64 Runtime.

### Windows: błąd ładowania biblioteki cuDNN

Jeśli karta NVIDIA jest wykrywana, ale test modeli kończy się błędem `Could not locate cudnn_engines_tensor_ir64_9.dll` lub `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`, problem dotyczy ładowania podbibliotek cuDNN. Aktualna wersja Wheelczura udostępnia katalogi bibliotek NVIDIA z wybranego środowiska i ładuje dodatkowe biblioteki przed uruchomieniem modelu. Nie trzeba ustawiać systemowej zmiennej `PATH`.

Zamknij launcher opcją **0**, rozpakuj aktualne pliki projektu do swojego folderu Wheelczura, a następnie sprawdź oba modele w PowerShell:

```powershell
.\.venv-cuda\Scripts\python.exe track_limb.py --check-cuda --device cuda:0
```

Jeśli test się powiedzie, uruchom analizę:

```powershell
.\.venv-cuda\Scripts\python.exe launcher.py --run cuda
```

Jeśli błąd nadal wskazuje brak tego pliku, sprawdź, czy pakiet go zawiera:

```powershell
Test-Path .\.venv-cuda\Lib\site-packages\nvidia\cudnn\bin\cudnn_engines_tensor_ir64_9.dll
```

Dla cuDNN **9.28.1.4** z opisanego komunikatu brakujący lub uszkodzony pakiet można przywrócić poleceniem:

```powershell
.\.venv-cuda\Scripts\python.exe -m pip install --force-reinstall --no-deps "nvidia-cudnn-cu12==9.28.1.4"
```

Jeśli masz inną wersję, sprawdź ją poleceniem `.\.venv-cuda\Scripts\python.exe -m pip show nvidia-cudnn-cu12` i użyj jej numeru zamiast `9.28.1.4`. Po naprawie ponów test `--check-cuda`. Nie kopiuj pojedynczych bibliotek DLL z innej wersji cuDNN.

Dokumentacja: [ONNX Runtime — ładowanie bibliotek CUDA/cuDNN](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls) oraz [NVIDIA — instalacja cuDNN w Windows](https://docs.nvidia.com/cudnn/installation/backend/windows/).

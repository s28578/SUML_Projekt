import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
from numpy.random import default_rng as rng

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()


st.sidebar.title("Panel boczny")
selected_option = st.sidebar.selectbox("Wybierz podstronę",
["Strona Główna", "Czym są słuchotki?", "Predykcja wieku słuchotki", "Informacje o danych", "Quiz o słuchotkach"])
if selected_option == "Strona Główna":


    st.markdown("<h1 style='text-align: center;'>Strona główna</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Twórcy</h2>", unsafe_allow_html=True)
    with st.container(border=True, horizontal_alignment="center", gap="small"):
        st.write("<div style='text-align: center;'>Aleksandra Kobek</div>", unsafe_allow_html=True)
        st.write("<div style='text-align: center;'>Daniel Krzemiński 🤡</div>", unsafe_allow_html=True)
        st.write("<div style='text-align: center;'>Kamila Litwin</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Cel projektu</h2>", unsafe_allow_html=True)
    st.write("Projekt został przygotowany na przedmiot \"Środowiska "
             "Uruchomieniowe AutoML\". Ma na celu przewidywanie wieku słuchotki na podstawie pomiarów fizycznych.")

    st.write("Wiek słuchotki określa się poprzez przecięcie muszli przez stożek, zabarwienie jej i zliczenie liczby "
             "pierścieni pod mikroskopem – jest to nudne i czasochłonne zadanie. "
             "W naszym projekcie do przewidywania wieku wykorzystuje się inne, łatwiejsze do uzyskania pomiary. "
             "Do w pełni wairygodnych wyników mogą być potrzebne dodatkowe informacje, takie jak "
             "warunki pogodowe i lokalizacja (a tym samym dostępność pożywienia).")

    st.write("W kolejnych stronach dostępne są informacje na temat słuchotek i samego modelu, a także możliwość "
             "przetestowania go przez wpisanie własnych danych.")

    # st.image("g.jpg")

elif selected_option == "Czym są słuchotki?":
    st.title("Czym są słuchotki?")
    # st.image("b.jpg")
    st.write("Słuchotki (Haliotidae) – rodzina ślimaków "
             "morskich.Słuchotki, nazywane też uchowcami, należą do jedynego rodzaju tej rodziny – Haliotis. Liczy on ponad 80 gatunków. "
             "Muszle mają różne rozmiary – od małych do dużych, mogą być okrągłe lub owalne. Skrętka jest zredukowana i spłaszczona. "
             "Ostatni skręt jest duży, kształtu spodka lub małżowiny usznej. Powierzchnia zewnętrzna pokryta nierównymi osiowymi lub "
             "spiralnymi liniami, żeberkami albo fałdami. Wzdłuż lewego brzegu ostatniego skrętu ciągnie się rząd okrągłych lub owalnych "
             "otworów; niektóre, wcześniejsze z nich mogą ulec zasklepieniu. Wewnętrzna powierzchnia muszli wyłożona jest opalizującą masą "
             "perłową, często z umieszczoną centralnie szeroką blizną mięśniową (miejsce, do którego przytwierdza się noga ślimaka). "
             "Brzeg wrzeciona pogrubiony i spłaszczony. Brak wieczka. Indianie pacyficznych wybrzeży Ameryki Północnej cenili muszle uchowców ze "
             "względu na piękny kolor masy perłowej i stosowali je zarówno jako surowiec do wyrobu biżuterii, jak i do inkrustacji.Wszystkie gatunki "
             "tej rodziny są roślinożerne. Żywią się głównie algami. Dorosłe osobniki praktycznie nie opuszczają raz wybranego miejsca. Żerują na tym"
             " samym obszarze przez całe życie. Bytują przytwierdzone do skalistego podłoża silną nogą, która jest cenionym przysmakiem "
             "kulinarnym (w kuchniach świata uchowce znane są jako tzw. abalony – wykwintna i droga potrawa). Po obu stronach potężnej nogi występuje"
             " fałd płaszczowy (epipodium), od którego odchodzi duża liczba brodawek czuciowych lub filamentów, dzięki którym ślimaki sprawiają wrażenie owłosionych. ")
    st.header("Występowanie i rozmnażanie")
    st.write("Zamieszkują głównie płytkie wody, ale niektóre osobniki można spotkać na głębokości nawet 400 m. Są rozdzielnopłciowe, z gonadami żeńskimi barwy zielonej i "
              "męskimi – żółtawej. Rozwój można prześledzić na podstawie kalifornijskiego gatunku Haliotis rufescens. Zapłodnione jaja powstają na wiosnę po wcześniejszym "
              "wyrzuceniu do wody niezapłodnionych komórek jajowych i spermy (zapłodnienie zewnętrzne). Dziesiątego dnia wolno pływająca larwa – weliger osiada na dno i po "
              "około 2 miesiącach rozwija się z niej miniaturka dorosłego osobnika. W wieku 1 roku ślimak osiąga wielkość około 2 cm, a po 4 latach uzyskuje dojrzałość płciową,"
              " mierząc około 12 cm. Wielkość konsumpcyjną osiągają po 15–20 latach. ")
    st.header("Galeria")
    tab1, tab2, tab3 = st.tabs(["Zdjęcie 1", "Zdjęcie 2",
                                "Zdjęcie 3"])
    with tab1:
        # st.image("c.jpg")
        st.write("Słuchotki na przybrzeżnych skałach podczas odpływu")
    with tab2:
        # st.image("d.jpg")
        st.write("Słuchotka podczas żerowania")
    with tab3:
        # st.image("e.jpg")
        st.write("Słuchotka kamczacka (Haliotis kamtschatkana)")
        # st.audio("bfg.mp3")


elif selected_option == "Predykcja wieku słuchotki":
    st.title("Predykcja wieku słuchotki")
    st.header("Podaj dane słuchotki: ")

    sex = st.radio("Sex:", ["M", "F","I"])
    length = st.number_input("Length:", min_value=0.0, format="%.3f")
    diameter = st.number_input("Diameter:", min_value=0.0, format="%.3f")
    height = st.number_input("Height:", min_value=0.0, format="%.3f")
    whole_weight = st.number_input("Whole weight:", min_value=0.0, format="%.4f")
    shucked_weight = st.number_input("Shucked weight:", min_value=0.0, format="%.4f")
    viscera_weight = st.number_input("Viscera weight:", min_value=0.0, format="%.4f")
    shell_weight = st.number_input("Shell weight:", min_value=0.0, format="%.3f")

    if st.button("Sprawdź wiek"):

        sex_encoded = {"M": 0, "F": 1, "I": 2}[sex]

        shucked_weight_proportion = shucked_weight / whole_weight
        viscera_weight_proportion = viscera_weight / whole_weight
        shell_weight_proportion = shell_weight / whole_weight


        input_data = np.array([[
            sex_encoded,
            length,
            diameter,
            height,
            whole_weight,
            shucked_weight,
            viscera_weight,
            shell_weight,
            shucked_weight_proportion,
            viscera_weight_proportion,
            shell_weight_proportion
        ]])

        prediction = model.predict(input_data)

        age = prediction[0] + 1.5

        st.success(f"Przewidywany wiek słuchotki: **{age:.1f} lat**")

elif selected_option == "Informacje o danych":

    st.markdown("<h1 style='text-align: center;'>Informacje o danych użytych do trenowania modelu</h1>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Informacje o cechach", "Heatmap",
                                "Rozkład cech", "Histogram"])
    with tab1:
        st.markdown("<h1 style='text-align: center;'>Informacje o cechach</h1>", unsafe_allow_html=True)
        dane = {
            "Variable name": ["Płeć", "Długość", "Średnica", "Wysokość", "Całkowita wysokość", "Waga po wyłuskaniu",
                              "Waga trzewi", "Waga muszli", "Pierścienie"],
            "Rola": ["Cecha", "Cecha", "Cecha", "Cecha", "Cecha", "Cecha", "Cecha", "Cecha", "Cel"],
            "Typ": ["Kategoryczny", "Ciągły", "Ciągły", "Ciągły", "Ciągły", "Ciągły", "Ciągły", "Ciągły",
                    "Liczba całkowita"],
            "Opis": ["M (samiec), F (samica), I (nowonarodzone)", "Najdłuższy wymiar muszli", "Prostopadle do długości",
                     "Z mięsem w skorupie", "Waga całej słuchotki", "Waga mięsa", "Waga trzewi (po wykrwawieniu)",
                     "Po wysuszeniu", "+1.5 daje wiek w latach"],

        }
        st.table(dane)
        st.markdown("<h3 style='text-align: center;'>Dane dodane na potrzeby projektu</h3>", unsafe_allow_html=True)
        dane2 = {
            "Variable name": ["Proporcja wagi wyłuskanej", "Proporcja wagi trzewi", "Proporcja wagi muszli"],
            "Rola": ["Cecha", "Cecha", "Cecha"],
            "Typ": ["Ciągły", "Ciągły", "Ciągły"],
            "Opis": ["Proporcja wagi wyłuskanej do całkowitej wagi, w zakresie 0-1",
                     "Proporcja wagi trzewi do całkowitej wagi, w zakresie 0-1",
                     "Proporcja wagi muszli do całkowitej wagi, w zakresie 0-1"],
        }
        st.table(dane2)
    with tab2:

        st.write("Miejsce na heatmap")
        with st.container(border=True):
            losowy_df = pd.DataFrame(rng(0).standard_normal((20, 3)), columns=["a", "b", "c"])
            st.bar_chart(losowy_df)
            st.caption("Tymczasowy wykres")
    with tab3:

        st.write("Miejsce na diagramy rozkładu poszczególnych cech")
        with st.container(border=True):
            losowy_df = pd.DataFrame(rng(1).standard_normal((20, 3)), columns=["a", "b", "c"])
            st.area_chart(losowy_df)
            st.caption("Tymczasowy wykres")
        tabb1, tabb2, tabb3, tabb4, tabb5, tabb6, tabb7, tabb8 = st.tabs(["Płeć", "Długość",
                                          "Średnica", "Wysokość", "Całkowita wysokość", "Waga po wyłuskaniu",
                              "Waga trzewi", "Waga muszli"])
        with tabb1:
            st.write("Płeć")
        with tabb2:

            st.write("Długość")
        with tabb3:

            st.write("Średnica")

        with tabb4:
            st.write("Wysokoś")

        with tabb5:
            st.write("Całkowita wysokość")
        with tabb6:
            st.write("Waga po wyłuskaniu")
        with tabb7:
            st.write("Waga trzewi")
        with tabb8:
            st.write("Waga muszli")

    with tab4:
        st.write("Miejsce na histogram")
        with st.container(border=True):
            losowy_df = pd.DataFrame(rng(2).standard_normal((20, 3)), columns=["a", "b", "c"])
            st.line_chart(losowy_df)
            st.caption("Tymczasowy wykres")



elif selected_option == "Quiz o słuchotkach":
    st.title("Quiz o ślimakach morskich")
    st.write("Sprawdź, ile wiesz o tych fascynujących stworzeniach!")

    questions = [
        {
            "q": "1. Jak nazywa się grupa bardzo kolorowych ślimaków morskich?",
            "options": ["Nudibranchia (nagoskrzelne)", "Patellogastropoda", "Neogastropoda", "Opisthobranchia"],
            "answer": "Nudibranchia (nagoskrzelne)"
        },
        {
            "q": "2. Jak ślimaki nagoskrzelne najczęściej bronią się przed drapieżnikami?",
            "options": ["Ukrywają się w muszli", "Wytwarzają toksyny", "Udają martwe", "Szybko pływają"],
            "answer": "Wytwarzają toksyny"
        },
        {
            "q": "3. Co zazwyczaj jedzą ślimaki nagoskrzelne?",
            "options": ["Ryby", "Koralowce, gąbki i parzydełkowce", "Plankton", "Glony"],
            "answer": "Koralowce, gąbki i parzydełkowce"
        },
        {
            "q": "4. Jak nazywa się słynny gatunek ślimaka morskiego znany z niebieskiego koloru i „skrzydełek”?",
            "options": ["Glaucus atlanticus", "Aplysia californica", "Hexabranchus sanguineus", "Elysia chlorotica"],
            "answer": "Glaucus atlanticus"
        },
        {
            "q": "5. Czym wyróżnia się Elysia chlorotica?",
            "options": ["Ma jedną z największych muszli", "Żyje w głębinach >3000 m", "Wykorzystuje fotosyntezę",
                        "Ma żuwaczki jak krab"],
            "answer": "Wykorzystuje fotosyntezę"
        }
    ]

    st.subheader("Pytania:")

    score = 0

    with st.form("quiz_form"):
        answers = []
        for i, q in enumerate(questions):
            st.write(q["q"])
            user_answer = st.radio("", q["options"], key=f"q{i}")
            answers.append(user_answer)

        submitted = st.form_submit_button("Sprawdź odpowiedzi")

    if submitted:
        for i, user_answer in enumerate(answers):
            if user_answer == questions[i]["answer"]:
                score += 1

        st.success(f"Twój wynik: **{score} / {len(questions)}**")

        if score == 5:
            st.balloons()
            # st.video("f.mp4")
            st.write("Perfekcyjnie! Znasz się na ślimakach morskich!")
        elif score >= 3:
            st.write("Całkiem nieźle!")
        else:
            st.write("Warto poczytać więcej o ślimakach morskich")

import ComplexPygame as C
import math
import cmath
import Color

def draw_triangle_colorat():
    # Setare dimensiuni ecran
    R = 6
    C.setXminXmaxYminYmax(-R, R, -R, R)

    # Definire vârfuri triunghi
    A = cmath.rect(4, math.radians(100))  # Punct A la 100°
    B = cmath.rect(3.5, math.radians(-30))  # Punct B la -30°
    C_ = cmath.rect(4, math.radians(220))  # Punct C la 220°

    # Funcție pentru a calcula direcția bisectoarei
    def bisector_direction(P1, P2, P3):
        v1 = P2 - P1
        v2 = P3 - P1
        bisec_dir = (v1 / abs(v1) + v2 / abs(v2))
        return bisec_dir / abs(bisec_dir)  # Normalizăm

    # Calculăm direcțiile bisectoarelor
    bis_dir_A = bisector_direction(A, B, C_)
    bis_dir_B = bisector_direction(B, A, C_)
    bis_dir_C = bisector_direction(C_, A, B)

    # Funcție pentru a determina pe ce parte a unei drepte se află un punct
    def which_side(point, line_point, line_direction):
        # Calculăm produsul vectorial pentru a determina partea
        to_point = point - line_point
        cross = (to_point.real * line_direction.imag - to_point.imag * line_direction.real)
        return cross > 0

    # Colorăm fiecare pixel în funcție de regiunea în care se află
    for z in C.screenAffixes():
        # Determinăm pe ce parte a fiecărei bisectoare se află punctul z
        side_A = which_side(z, A, bis_dir_A)
        side_B = which_side(z, B, bis_dir_B)
        side_C = which_side(z, C_, bis_dir_C)

        # Codificăm regiunea folosind un număr de la 0 la 7 (3 biți pentru 3 bisectoare)
        region = (side_A << 2) + (side_B << 1) + side_C

        # Atribuim culori diferite pentru fiecare regiune
        colors = [
            Color.Lightgray,  # 000
            Color.Lightblue,  # 001
            Color.Lightgreen,  # 010
            Color.Lightcyan,  # 011
            Color.Lightpink,  # 100
            Color.Lightyellow,  # 101
            Color.Wheat,  # 110
            Color.Lavender  # 111
        ]

        C.setPixel(z, colors[region])

    # Desenăm doar triunghiul
    C.drawLine(A, B, Color.Black)
    C.drawLine(B, C_, Color.Black)
    C.drawLine(C_, A, Color.Black)



def triangle_circumcircle_regions():
    # Setare dimensiuni ecran
    R = 6
    C.setXminXmaxYminYmax(-R, R, -R, R)

    # Fixăm cercul circumscris
    centru_cerc = complex(0, 0)  # Centrul cercului în origine
    raza_cerc = 3.5  # Raza cercului

    # Plasăm cele 3 puncte pe cerc
    A = centru_cerc + cmath.rect(raza_cerc, math.radians(90))  # 90°
    B = centru_cerc + cmath.rect(raza_cerc, math.radians(210))  # 210°
    C_ = centru_cerc + cmath.rect(raza_cerc, math.radians(330))  # 330°

    # Funcție pentru a determina pe ce parte a unei drepte se află un punct
    def which_side_of_line(point, line_start, line_end):
        # Calculăm produsul vectorial pentru a determina partea
        # Pozitiv = stânga, Negativ = dreapta
        v1 = line_end - line_start
        v2 = point - line_start
        cross = v1.real * v2.imag - v1.imag * v2.real
        return cross > 0

    # Funcție pentru a verifica dacă un punct este în interiorul cercului
    def inside_circle(point, center, radius):
        return abs(point - center) <= radius

    # Colorăm fiecare pixel în funcție de regiunea în care se află
    for z in C.screenAffixes():
        # Determinăm pe ce parte a fiecărei laturi se află punctul z
        side_AB = which_side_of_line(z, A, B)  # Latura AB
        side_BC = which_side_of_line(z, B, C_)  # Latura BC
        side_CA = which_side_of_line(z, C_, A)  # Latura CA

        # Determinăm dacă punctul este în interiorul cercului
        inside_circ = inside_circle(z, centru_cerc, raza_cerc)

        # Identificăm regiunea corectă (doar 10 regiuni geometrice posibile)
        if inside_circ:
            # În interiorul cercului
            if side_AB and side_BC and side_CA:
                region = 0  # Interior triunghi (este și în cerc)
            elif side_AB and side_BC and not side_CA:
                region = 1  # Segment circular între A și B
            elif side_BC and side_CA and not side_AB:
                region = 2  # Segment circular între B și C
            elif side_CA and side_AB and not side_BC:
                region = 3  # Segment circular între C și A
            else:
                region = 0  # Alte cazuri în cerc -> interior triunghi
        else:
            # În exteriorul cercului - 6 regiuni
            if side_AB and side_BC and side_CA:
                region = 4  # Regiunea "în fața" triunghiului
            elif side_AB and side_BC and not side_CA:
                region = 5  # Regiunea "în fața" laturii AB
            elif side_BC and side_CA and not side_AB:
                region = 6  # Regiunea "în fața" laturii BC
            elif side_CA and side_AB and not side_BC:
                region = 7  # Regiunea "în fața" laturii CA
            elif side_AB and not side_BC and not side_CA:
                region = 8  # Regiunea "în spatele" vârfului C
            elif not side_AB and side_BC and not side_CA:
                region = 9  # Regiunea "în spatele" vârfului A
            elif not side_AB and not side_BC and side_CA:
                region = 4  # Regiunea "în spatele" vârfului B -> mapată la regiunea 4
            else:
                region = 4  # Alte cazuri -> mapate la regiunea 4

        # Atribuim culori diferite pentru cele 10 regiuni
        colors = [
            Color.Darkgreen,  # 0 - Interior triunghi
            Color.Turquoise,  # 1 - Segment circular AB
            Color.Blue,  # 2 - Segment circular BC
            Color.Lightblue,  # 3 - Segment circular CA
            Color.Purple,  # 4 - Exterior cerc, "în fața" triunghiului
            Color.Mediumpurple,  # 5 - Exterior cerc, "în fața" AB
            Color.Violet,  # 6 - Exterior cerc, "în fața" BC
            Color.Yellow,  # 7 - Exterior cerc, "în fața" CA
            Color.Lightyellow,  # 8 - Exterior cerc, "în spatele" C
            Color.Cornflowerblue  # 9 - Exterior cerc, "în spatele" A
        ]

        C.setPixel(z, colors[region])

    # Actualizăm ecranul
    C.refreshScreen()

####################################################################################TEMA 5
def recDraw(a, b, c, index):
    if index <= 20:
        recDraw(a, b, c, index + 1)
    C.drawNgon([a, b, c], Color.Navy)

    # C.drawNgon([(a*4/5+b*1/5), (b*4/5+c*1/5), (c*4/5+a*1/5)], Color.Navy)


def Mijloace():
    C.fillScreen()
    C.setXminXmaxYminYmax(0, 10, 0, 10)
    a = 1 + 1j
    b = 9 + 2j
    c = 5 + 9j
    col = Color.Navy

    for i in range(0, 20):
        C.drawNgon([(a), (b), (c)], col)
        aa = a
        bb = b
        cc = c
        a = aa * 4 / 5 + bb * 1 / 5
        b = bb * 4 / 5 + cc * 1 / 5
        c = cc * 4 / 5 + aa * 1 / 5


def prob2():
    C.setXminXmaxYminYmax(-10, 10, -10, 10)
    a = 0 + 1j
    b = 1 / 2 + 0j
    c = -1 / 2 + 0j
    aa = a
    bb = b
    cc = c
    for i in range(0, 12):
        C.drawNgon([a, b, c], Color.Navy)
        a = aa * 1.12 - cc * 0.20
        b = bb * 1.12 - aa * 0.20
        c = cc * 1.12 - bb * 0.20
        aa = a
        bb = b
        cc = c
    C.drawNgon([a, b, c], Color.Navy)

###############################################################################################TEMA 6
def vf_120(A, B, opposite):
    mij = (A + B) / 2
    v = B - A
    L = abs(v)
    h = L / (2 * math.sqrt(3))  # inaltimea de la mijl bazei la vf
    u = v * 1j / abs(v)  # vectorul unitar perp pe AB

    # calc produsul scalar pentru a decide ce parte este interiorul lui ABC.
    cp = (B - A).real * (opposite - A).imag - (B - A).imag * (opposite - A).real
    # daca interiorul este la stanga lui AB (cp > 0), atunci externalul se construiește pe dreapta lui AB.
    if cp > 0:
        return mij - h * u
    else:
        return mij + h * u


def desenare():
    C.setXminXmaxYminYmax(-0.5, 1.2, -0.8, 0.8)
    C.fillScreen(Color.Whitesmoke)

    zA = 0 + 0j
    zB = 1 + 0j
    b = math.sin(math.radians(50)) / math.sin(math.radians(100))
    zC = b * (math.cos(math.radians(30)) + 1j * math.sin(math.radians(30)))

    # triunghiul ABC
    C.drawNgon([zA, zB, zC], Color.Black)
    C.setText("A", zA, Color.Black, 16)
    C.setText("B", zB, Color.Black, 16)
    C.setText("C", zC, Color.Black, 16)

    # Construim triunghiurile isoscele externe cu vârful de 120° pe fiecare latură:

    # Pe latura AB – vârful D; punct opus: C
    zD = vf_120(zA, zB, zC)
    C.drawNgon([zA, zB, zD], Color.Red)

    # Pe latura BC – vârful E; punct opus: A
    zE = vf_120(zB, zC, zA)
    C.drawNgon([zB, zC, zE], Color.Green)

    # Pe latura CA – vârful F; punct opus: B
    zF = vf_120(zC, zA, zB)
    C.drawNgon([zC, zA, zF], Color.Blue)

    Aprim = zE #A' = zE
    Bprim = zF #B' = zF
    Cprim = zD #C' = zD

    # Desenăm triunghiul A'B'C'
    C.drawNgon([Aprim, Bprim, Cprim], Color.Purple)
    C.setText("A'", Aprim, Color.Black, 16)
    C.setText("B'", Bprim, Color.Black, 16)
    C.setText("C'", Cprim, Color.Black, 16)

    C.refreshScreen()


def bazaUnghiUnghi(zB, zC, uB, uC, peStg=True):
    """
    Determină vârful A al unui triunghi dreptunghic cu bază [zB, zC] și cu
    unghiurile la B și C: uB și uC (în radiani), având uB + uC = 90°.

    Deoarece zC este fix (originea) și d = zC - zB, avem:
         |BC| = |zC - zB|
    iar formula derivată din teorema sinuselor (pentru uB+uC = 90°)
         |AB| = |BC| * sin(uC)
    se folosește pentru a calcula lungimea segmentului AB.

    Direcția de la zB către A se obține prin:
         A = zB + |AB| * exp(i*(arg(zC - zB) + uB))
    (pentru soluția „pe stânga” – pentru soluția oglindită s-ar folosi -uB).
    """
    if not peStg:
        uB, uC = -uB, -uC
    d = zC - zB
    L = abs(d)
    # Deoarece sin(uB+uC) = sin(90°) = 1, avem:
    AB = L * math.sin(uC)
    base_angle = math.atan2(d.imag, d.real)
    angle = base_angle + uB  # alegem soluția "pe stânga"
    A = zB + AB * complex(math.cos(angle), math.sin(angle))
    return A


def desenare2():
    """
    Construcţia se bazează pe triunghiuri similare 30–60–90 cu
    ipotenuza la origine. Fiecare triunghi este dat de
      - baza: segmentul dintre un vârf din spirală (pe care îl notăm V_n)
        și originea (0) – aceasta este comună tuturor triunghiurilor, dar
        0 NU face parte din spirală,
      - vârful de tranziţie: al doilea vârf V_{n+1} care (după calcul)
        se va așeza exact pe una dintre axe.

    Alegem ca spirală de puncte:
      • V_0 pe Ox (prin impunere; vom alege un V_0 negativ),
      • V_1 pe Oy,
      • V_2 pe Ox,
      • V_3 pe Oy etc.

    Relaţia de scară, din triunghiul 30–60–90, este
         |OV_{n+1}| = |OV_n|/√3.
    Alegem, de asemenea, semnele astfel încât alternanța să fie:
         V_0 = (–a, 0),
         V_1 = (0, +a/√3),
         V_2 = (+a/(√3)², 0),
         V_3 = (0, –a/(√3)³),
         V_4 = (–a/(√3)⁴, 0), etc.
    """
    # Stabilim o fereastră suficient de mare
    C.setXminXmaxYminYmax(-8, 13, -12, 12)
    C.fillScreen(Color.Whitesmoke)

    a = 12.0  # lungimea inițială (alegeți o valoare mare pentru o spirală vizibilă)
    n_tri = 10  # numărul de triunghiuri (iar, deci, numărul de puncte din spirală va fi n_tri+1)

    spiral_points = []
    # Calculăm punctele spirală V_0, V_1, …, V_n_tri (conform regulii de mai sus)
    for idx in range(n_tri + 1):
        if idx % 2 == 0:
            # Indicii pari: punctul trebuie să fie pe Ox.
            # Semnul se alternează: V_0 negativ, V_2 pozitiv, V_4 negativ, …
            sign = (-1) ** (idx // 2)
            x = sign * a / (math.sqrt(3) ** idx)
            y = 0
            V = complex(x, y)
        else:
            # Indicii impar: punctul trebuie să fie pe Oy.
            # Semnul se alternaă: V_1 pozitiv, V_3 negativ, V_5 pozitiv, …
            sign = (-1) ** ((idx - 1) // 2)
            x = 0
            y = sign * a / (math.sqrt(3) ** idx)
            V = complex(x, y)
        spiral_points.append(V)

    # Pentru fiecare triunghi cu bază [V_n, 0] și vârf = V_{n+1}, desenăm conturul în roșu.
    for i in range(len(spiral_points) - 1):
        tri_points = [spiral_points[i], 0 + 0j, spiral_points[i + 1]]
        C.drawNgon(tri_points, Color.Red)

    # Desenăm cu o linie albastră punctele spirală conectate (acesta este conturul spirală)
    for i in range(len(spiral_points) - 1):
        C.drawLine(spiral_points[i], spiral_points[i + 1], Color.Blue)

    C.refreshScreen()


###############################################################################################TEMA 7

def unCercQR(q, r, N):
    alfa = 2 * math.pi / N
    return [q + C.fromRhoTheta(r, k * alfa) for k in range(N)]


def Mediatoare():
    C.setXminXmaxYminYmax(0, 10, 0, 10)
    zA = 2 + 2.5j
    zB = 8 + 2.5j
    zC = 3.5 + 9j

    sq_len_a = abs(zB - zC) ** 2
    sq_len_b = abs(zC - zA) ** 2
    sq_len_c = abs(zA - zB) ** 2

    alpha = sq_len_a * (sq_len_b + sq_len_c - sq_len_a)
    beta = sq_len_b * (sq_len_c + sq_len_a - sq_len_b)
    gamma = sq_len_c * (sq_len_a + sq_len_b - sq_len_c)

    denominator = alpha + beta + gamma
    C.setAxis()
    z0 = (alpha * zA + beta * zB + gamma * zC) / denominator

    for z in C.screenAffixes():
        za = C.rho(z - zA)
        zb = C.rho(z - zB)
        zc = C.rho(z - zC)
        k = 0
        if za < zb:
            k += 1
        if zb < zc:
            k += 2
        if zc < za:
            k += 4
        C.setPixel(z, Color.Index(600 + 50 * k))
    C.drawLine(z0, zA, Color.Black)
    C.drawLine(z0, zB, Color.Black)
    C.drawLine(z0, zC, Color.Black)
    C.drawNgon([zA, zB, zC], Color.Red)
    C.drawNgon(unCercQR(z0, abs(z0 - zA), 1000), Color.Red)


###################################################################


def LocGeomI():
    def cercInscris(zA, zB, zC):
        # returneaza zI si r pentru cercul inscris
        a = C.rho(zB - zC)
        b = C.rho(zC - zA)
        c = C.rho(zA - zB)
        p = (a + b + c) / 2
        S = math.sqrt(p * (p - c) * (p - b) * (p - a))
        zI = (a * zA + b * zB + c * zC) / (a + b + c)
        r = S / p
        return zI, r

    C.setXminXmaxYminYmax(-10, 10, -10, 10)
    q = 0
    R = 7
    nrPuncte = 720
    delta = 2 * math.pi / nrPuncte
    nB = nrPuncte // 2 + nrPuncte // 15
    nC = nrPuncte - nrPuncte // 15
    zB = C.fromRhoTheta(R, nB * delta)
    zC = C.fromRhoTheta(R, nC * delta)

    for n in range(10 * nrPuncte):
        C.fillScreen(Color.White)
        C.setNgon(unCercQR(q, R, nrPuncte), Color.Navy)
        zA = C.fromRhoTheta(R, n * delta)
        C.drawNgon([zA, zB, zC], Color.Navy)
        zI, r = cercInscris(zA, zB, zC)
        C.setNgon(unCercQR(zI, r, 300), Color.Green)
        C.drawNgon([zA, zI, zB, zI, zC, zI], Color.Green)
        if C.mustClose():
            return


###################################################################

def LocGeomH():
    def ortocentru(zA, zB, zC):

        sq_len_a = abs(zB - zC) ** 2
        sq_len_b = abs(zC - zA) ** 2
        sq_len_c = abs(zA - zB) ** 2

        alpha = (sq_len_c + sq_len_a - sq_len_b) * (sq_len_a + sq_len_b - sq_len_c)
        beta = (sq_len_a + sq_len_b - sq_len_c) * (sq_len_c + sq_len_b - sq_len_a)
        gamma = (sq_len_c + sq_len_b - sq_len_a) * (sq_len_b + sq_len_c - sq_len_a)
        if (alpha + beta + gamma != 0):
            zH = (alpha * zA + beta * zB + gamma * zC) / (alpha + beta + gamma)
        else:
            zH = 0
        return zH

    C.setXminXmaxYminYmax(-10, 10, -12, 8)
    q = 0
    R = 6
    nrPuncte = 720
    delta = 2 * math.pi / nrPuncte
    nB = nrPuncte // 2 + nrPuncte // 15
    nC = nrPuncte - nrPuncte // 15
    zB = C.fromRhoTheta(R, nB * delta)
    zC = C.fromRhoTheta(R, nC * delta)
    lista_puncte = []

    for n in range(10 * nrPuncte):

        if n % nrPuncte == nB or n % nrPuncte == nC:
            continue
        C.fillScreen(Color.White)
        for x in lista_puncte:
            C.setPixel(x, Color.Red)

        C.setNgon(unCercQR(q, R, nrPuncte), Color.Navy)
        zA = C.fromRhoTheta(R, n * delta)
        C.drawNgon([zA, zB, zC], Color.Navy)
        zH = ortocentru(zA, zB, zC)
        lista_puncte.append(zH)
        C.drawNgon([zA, zH, zB, zH, zC, zH], Color.Green)
        if C.mustClose():
            return

##################################################################################################TEMA 8

def HeptaPentagon2():
    def npGonQA(q, a0, n, p=1):
        theta = p * 2.0 * math.pi / n
        return [q + C.fromRhoTheta(1, k * theta) * (a0 - q) for k in range(n)]

    def bazaApex(zB, zC, uA, peStg=True):
        omegaA = C.fromRhoTheta(1, uA) if peStg else C.fromRhoTheta(1, -uA)
        zA = (zC - omegaA * zB) / (1 - omegaA)
        return zA

    def construiestePoligoanePeLaturi(poligoane, nExt, thetaExt, culoare):
        poligoaneNoi = []
        for poligon in poligoane:
            for k in range(len(poligon) - 1):
                qk = bazaApex(poligon[k], poligon[k + 1], thetaExt, False)
                pentagon = npGonQA(qk, poligon[k], nExt)
                pentagon.append(pentagon[0])
                C.drawNgon(pentagon[:-1], culoare)
                poligoaneNoi.append(pentagon)
        return poligoaneNoi

    def construiestePeVarfuriExterioareConsecutive(poligoane, nExt, thetaExt, culoare, centru=0):
        poligoaneNoi = []

        for i in range(len(poligoane) - 1):
            p1 = poligoane[i][:-1]  # fără duplicat
            p2 = poligoane[i + 1][:-1]

            # Cel mai exterior vârf din fiecare pentagon (față de centru)
            z1 = max(p1, key=lambda z: abs(z - centru))
            z2 = max(p2, key=lambda z: abs(z - centru))

            # Construim apexul între cele două vârfuri
            qk = bazaApex(z1, z2, thetaExt, peStg=False)

            # Construim pentagonul
            pentagon = npGonQA(qk, z1, nExt)
            pentagon.append(pentagon[0])

            C.drawNgon(pentagon[:-1], culoare)
            poligoaneNoi.append(pentagon)

        return poligoaneNoi

    C.setXminXmaxYminYmax(-5, 5, -5, 5)
    C.fillScreen()

    q = 0
    a = 2
    nInt = 7
    nExt = 5
    thetaExt = 2 * math.pi / nExt
    numarIteratii = 1

    pInt = npGonQA(q, a, nInt)
    pInt.append(pInt[0])
    C.drawNgon(pInt[:-1], Color.Black)

    poligoaneCurente = [pInt]
    culori = [Color.Red, Color.Green, Color.Purple, Color.Orange, Color.Yellow]

    # Primul rând de pentagoane: pe laturi
    culoare1 = culori[0 % len(culori)]
    poligoanel1 = construiestePoligoanePeLaturi(poligoaneCurente, nExt, thetaExt, culoare1)

    # Al doilea rând de pentagoane: pe vârfuri ale poligoanelor din rândul 1
    if len(poligoanel1) >= 2:
        construiestePeVarfuriExterioareConsecutive(poligoanel1, nExt, thetaExt, Color.Black)






def HeptaPentagon3():
    def npGonQA(q, a0, n, p=1):
        theta = p * 2.0 * math.pi / n
        return [q + C.fromRhoTheta(1, k * theta) * (a0 - q) for k in range(n)]

    def bazaApex(zB, zC, uA, peStg=True):
        omegaA = C.fromRhoTheta(1, uA) if peStg else C.fromRhoTheta(1, -uA)
        zA = (zC - omegaA * zB) / (1 - omegaA)
        return zA

    def construiestePoligoanePeLaturi(poligoane, nExt, thetaExt, culoare):
        poligoaneNoi = []
        for poligon in poligoane:
            for k in range(len(poligon) - 1):
                qk = bazaApex(poligon[k], poligon[k + 1], thetaExt, False)
                pentagon = npGonQA(qk, poligon[k], nExt)
                pentagon.append(pentagon[0])
                C.fillNgon(pentagon[:-1], culoare)
                poligoaneNoi.append(pentagon)
        return poligoaneNoi

    def construiestePoligoanePeVarfuriExterioare(petaleRand1, nExt, thetaExt, culoare, centru=0):
        poligoaneNoi = []

        for i in range(len(petaleRand1)):
            p1 = petaleRand1[i]
            p2 = petaleRand1[(i + 1)%len(petaleRand1)]

            # Găsește vârful cel mai îndepărtat (exterior) din fiecare petală
            z1 = max(p1[:-1], key=lambda z: abs(z - centru))
            z2 = max(p2[:-1], key=lambda z: abs(z - centru))

            # Calculăm apexul noii "petale"
            qk = bazaApex(z1, z2, thetaExt, peStg=False)

            # Construim pentagonul cu vârful între z1 și z2
            pentagon = npGonQA(qk, z1, nExt)
            pentagon.append(pentagon[0])

            C.fillNgon(pentagon[:-1], culoare)
            poligoaneNoi.append(pentagon)

        return poligoaneNoi

    # === Setup general ===
    C.setXminXmaxYminYmax(-5, 5, -5, 5)
    C.fillScreen()

    q = 0
    a = 2
    nInt = 7       # Heptagon central
    nExt = 5       # Pentagoane în jur
    thetaExt = 2 * math.pi / nExt

    # === Heptagon central ===
    pInt = npGonQA(q, a, nInt)
    pInt.append(pInt[0])

    # === Rândul 1: pentagoane pe laturile heptagonului ===
    culoareRand1 = Color.Navajowhite  # sau altă nuanță de "nude"
    petaleRand1 = construiestePoligoanePeLaturi([pInt], nExt, thetaExt, culoareRand1)

    # === Rândul 2: pentagoane pe vârfurile exterioare ale petalelor din rândul 1 ===
    culoareRand2 = Color.Lightpink
    construiestePoligoanePeVarfuriExterioare(petaleRand1, nExt, thetaExt, culoareRand2)

###########################################################################################TEMA 9

def PseudoSpiralaLuiArhimede():
    nrPuncte = 1000
    alfa = math.pi/2
    omega = C.fromRhoTheta(1, alfa / nrPuncte)

    def traseazaArc(q, delta):
        for k in range(nrPuncte):
            delta *= omega
            C.setPixel(q + delta, Color.Red)
        versor = delta / C.rho(delta)
        q -= versor
        delta += versor
        C.drawLine(q, q + delta, Color.Black)
        return q, delta

    lat = 20
    C.setXminXmaxYminYmax(-lat, lat, -lat, lat)
    q = 0
    delta = 1j
    for k in range(20):
        q, delta = traseazaArc(q, delta)
    C.refreshScreen()

def GoldenRatio2024():
    fi = (1 + math.sqrt(5.0)) / 2
    omegaCDprimPeAD = -1j / fi

    def traseazaSiTransforma(sector):
        a = sector[0]
        d = sector[- 1]
        c = sector[-2]
        b = a + c - d
        dprim = c + (d - c) / fi
        C.fillNgon([a, b, c, d], Color.Gold)
        # C.fillNgon(sector, Color.Red)
        C.drawNgon([a, b, c, d], Color.Red)
        return [dprim + omegaCDprimPeAD * (z - d) for z in sector]

    C.setXminXmaxYminYmax(-0.5, 2, -0.75, 1.75)
    C.fillScreen(Color.Mediumaquamarine)
    a = 0
    b = 1j
    c = 1 + 1j
    d = 1
    C.setText("A", a - 0.1j)
    C.setText("B", b + 0.01j)
    C.setText("C", c + 0.01j)
    C.setText("D", d - 0.1j)
    nrPuncte = 1000
    alfa = -math.pi / (2 * nrPuncte)
    sector = [d + C.fromRhoTheta(1, n * alfa) * (a - d) for n in range(nrPuncte)]
    sector.append(d)
    for k in range(10):
        sector = traseazaSiTransforma(sector)
        C.refreshScreen()
        C.wait(100)


############################################################################################TEMA 10

def vf_pentagon(centru=0j, raza=1.0):
    # generez vf unui pentagon cu raza data
    varfuri = []
    for k in range(5):
        angle = 2 * math.pi * k / 5 - math.pi / 2  # începem de sus
        vf = centru + raza * cmath.exp(1j * angle)
        varfuri.append(vf)
    return varfuri


def curba_koch_generalizata(start, fin, theta=math.pi / 3, lam=1.0 / 3, nrIter=15):
    # theta: unghiul de rotație pentru segmentul central
    # lam: proporția pentru împărțirea segmentului
    if nrIter == 0:
        return [start, fin]

    # punctele de diviziune
    dir = fin - start
    length = abs(dir)
    unit_dir = dir / length if length > 0 else 1

    # punctele A, B, C, D pentru construcția koch
    A = start
    B = start + lam * dir
    D = start + (1 - lam) * dir
    E = fin

    # punctul C se calculeaza rotind segmentul BD cu unghiul theta
    BD = D - B
    BD_rotit = BD * cmath.exp(1j * theta)
    C = B + BD_rotit

    # recursiv pentru fiecare segment
    points = []
    points.extend(curba_koch_generalizata(A, B, theta, lam, nrIter - 1)[:-1])
    points.extend(curba_koch_generalizata(B, C, theta, lam, nrIter - 1)[:-1])
    points.extend(curba_koch_generalizata(C, D, theta, lam, nrIter - 1)[:-1])
    points.extend(curba_koch_generalizata(D, E, theta, lam, nrIter - 1))

    return points


"""
def pentagonKoch():

    #Desenează pentagonul lui Koch cu curbe Koch pe fiecare latură
    C.setXminXmaxYminYmax(-2.5, 2.5, -2.5, 2.5)
    C.fillScreen(Color.White)

    # parametrii
    theta = math.pi / 3 
    lam = 1.0 / 3  
    nrIter = 4  

    # Generăm vârfurile pentagonului
    varfuri = vf_pentagon(centru=0j, raza=1.5)

    # Pentru fiecare latură a pentagonului, desenăm o curbă Koch
    colors = [Color.Red, Color.Blue, Color.Green, Color.Orange, Color.Purple]

    for i in range(5):
        vf_start = varfuri[i]
        vf_fin = varfuri[(i + 1) % 5]

        # Generăm curba Koch pentru această latură
        pct_koch = curba_koch_generalizata(vf_start, vf_fin, theta, lam, nrIter)

        # Desenăm curba punct cu punct
        color = colors[i % len(colors)]
        for j in range(len(pct_koch) - 1):
            C.drawLine(pct_koch[j], pct_koch[j + 1], color)

            # Verificăm dacă trebuie să oprim desenarea
            if C.mustClose():
                return
"""


def pentagon():
    # deseneaza un pentagon koch
    C.setXminXmaxYminYmax(-2.2, 2.2, -2.2, 2.2)
    C.fillScreen(Color.Whitesmoke)

    theta = math.pi / 3  # 60 de grade
    lam = 1.0 / 3
    nrIter = 4

    centru = 0j
    raza = 1.8

    varfuri = vf_pentagon(centru=centru, raza=raza)

    colors = [Color.Red, Color.Blue, Color.Green, Color.Orange, Color.Purple]

    for i in range(5):
        vf_start = varfuri[i]
        vf_fin = varfuri[(i + 1) % 5]

        # curba Koch pentru aceasta latura
        pct_koch = curba_koch_generalizata(vf_start, vf_fin, theta, lam, nrIter)

        # desenam curba
        color = colors[i]
        for j in range(len(pct_koch) - 1):
            C.drawLine(pct_koch[j], pct_koch[j + 1], color)

            if C.mustClose():
                return


################################################

def pentagon_baza(R):
    pent = [R * math.e ** (1j * (math.pi / 2 + 2 * math.pi * k / 5)) for k in range(5)]
    pent.append(pent[0])
    return pent


def poligon(varfuri, color):
    for i in range(1, len(varfuri)):
        C.drawLine(varfuri[i - 1], varfuri[i], color)


def Sierpinski(a, b, lvl, r, P0, adancime):
    # a, b - parametrii transformarii curente (T(z)= a*z+b);
    # lvl – adancimea recursiei;
    # r      – factorul de contracție
    # P0     – lista de vf a pentagonului de bază
    # adancime  – adâncimea actuală, its useful for color

    if lvl == 0:
        polig = [a * v + b for v in P0]
        # Se alege culoarea în funcție de adâncimea recursiei curente.
        poligon(polig, Color.Index((adancime + 100) % 5))
    else:
        for i in range(5):
            # T_new(z) = r*(a*z+b) + (1-r)*P0[i]
            anou = r * a
            bnou = r * b + (1 - r) * P0[i]
            Sierpinski(anou, bnou, lvl - 1, r, P0, adancime + 1)


def SierpinskiPentagon():
    C.setXminXmaxYminYmax(-1.5, 1.5, -1.5, 1.5)
    C.fillScreen()

    R = 1.0
    # baza = un pentagon regulat centrat la 0.
    P0 = pentagon_baza(R)

    # factorul de contracție
    r = (3 - math.sqrt(5)) / 2.0  # ≈ 0.382

    lvl_maxim = 4

    # T(z) = 1*z + 0.
    Sierpinski(1, 0, lvl_maxim, r, P0, 0)

    C.refreshScreen()
    while not C.mustClose():
        C.wait(50)

####################################################################################TEMA 11
def Peano():
    c1 = (1 + 1j) / 6
    c2 = (1 + 3j) / 6
    c3 = (1 + 5j) / 6
    c4 = (3 + 5j) / 6
    c0 = (3 + 3j) / 6
    c5 = (3 + 1j) / 6
    c6 = (5 + 1j) / 6
    c7 = (5 + 3j) / 6
    c8 = (5 + 5j) / 6

    def s0(z):
        return c0 - (z - c0) / 3

    def s1(z):
        return c1 + (z - c0) / 3

    def s2(z):
        return c2 - (z - c0).conjugate() / 3

    def s3(z):
        return c3 + (z - c0) / 3

    def s4(z):
        return c4 + (z - c0).conjugate() / 3

    def s5(z):
        return c5 + (z - c0).conjugate() / 3

    def s6(z):
        return c6 + (z - c0) / 3

    def s7(z):
        return c7 - (z - c0).conjugate() / 3

    def s8(z):
        return c8 + (z - c0) / 3

    def transforma(li):
        rez = []
        for z in li:  rez.append(s1(z))
        for z in li:  rez.append(s2(z))
        for z in li:  rez.append(s3(z))
        for z in li:  rez.append(s4(z))
        for z in li:  rez.append(s0(z))
        for z in li:  rez.append(s5(z))
        for z in li:  rez.append(s6(z))
        for z in li:  rez.append(s7(z))
        for z in li:  rez.append(s8(z))
        return rez

    def traseaza(li):
        C.fillScreen()
        # trasam chenarul
        C.drawNgon([0, 1, 1 + 1j, 1j], Color.Black)
        for n in range(1, len(li)):
            col = Color.Red if n % 9 == 0 else Color.Blue
            C.drawLine(li[n - 1], li[n], col)

    C.setXminXmaxYminYmax(-0.1, 1.1, -0.1, 1.1)
    fig = [c0]
    for k in range(2):
        fig = transforma(fig)
        traseaza(fig)
        if C.mustClose(): return




##########################################################
def KochCu2Transformari():
    theta = math.pi / 6
    rho = 0.5 / math.cos(theta)
    w = C.fromRhoTheta(rho, theta)
    zA = 0
    zB = 1
    zC = zA + w * (zB - zA)
    omega1 = (zC - zA) / (zB - zA).conjugate()
    omega2 = (zC - zB) / (zA - zB).conjugate()

    def T1(z):
        return zA + omega1 * (z - zA).conjugate()

    def T2(z):
        return zB + omega2 * (z - zB).conjugate()

    def transforma(li):
        rez = [T1(z) for z in li]
        rez.extend([T2(z) for z in li])
        return rez

    C.setXminXmaxYminYmax(-0.1, 1.1, -0.1, 1.1)
    fig = [zA, zB]
    nrEtape = 10
    for k in range(nrEtape):
        fig = transforma(fig)
        C.fillScreen()
        col = Color.Index(300 + 10 * k)
        for h in range(1, len(fig)):
            C.drawLine(fig[h - 1], fig[h], col)
        if C.mustClose(): return
        C.wait(50)
    C.setAxis()

###############################################################
def SierpinskiTriunghiDrepunghic():
    zB = 0
    zC = 1
    zQ = (zB + zC) / 2
    zA = zQ + C.fromRhoTheta(C.rho(zC - zQ), 2 * math.pi / 5)
    k1 = (zA - zC) / (zB - zC).conjugate()
    k2 = (zA - zB) / (zC - zB).conjugate()

    def s1(z):
        return zC + k1 * (z - zC).conjugate()

    def s2(z):
        return zB + k2 * (z - zB).conjugate()

    def transforma(li):
        rez = []
        for z in li:
            rez.append(s1(z))
        for z in li:
            rez.append(s2(z))
        return rez

    def traseaza(li):

        C.fillScreen()
        z1 = li[0]
        z2 = li[1]
        z3 = li[2]
        zA = (z1 + z2 + z3) / 3
        C.drawLine(z1, z2, Color.Blue)
        C.drawLine(z2, z3, Color.Blue)
        C.drawLine(z3, z1, Color.Blue)
        if (len(li) == 3): return
        for n in range(5, len(li), 3):
            z1 = li[n - 2]
            z2 = li[n - 1]
            z3 = li[n]
            zB = (z1 + z2 + z3) / 3
            C.drawLine(z1, z2, Color.Blue)
            C.drawLine(z2, z3, Color.Blue)
            C.drawLine(z3, z1, Color.Blue)
            C.drawLine(zA, zB, Color.Red)
            zA = zB

    #   makeImage()
    #    {
    C.setXminXmaxYminYmax(-0.1, 1.1, -0.1, 1.1)
    fig = [zB, zA, zC]

    # traseaza(fig);
    for k in range(7):
        fig = transforma(fig)
        traseaza(fig)
        if C.mustClose(): return
    return

##############################################################

def HilberTriunghi():
    class Triunghi:
        def __init__(self, a, b, c):
            self.a, self.b, self.c = a, b, c

        def show(self, col):
            C.fillNgon([self.a, self.b, self.c], col)

        def centru(self):
            return (self.a + self.b + self.c ) / 3

    def transforma(li):
        rez = []
        for P in li:
            mab, mbc, mac = (P.a + P.b) / 2, (P.b + P.c) / 2, (P.c + P.a) / 2
            c0 = P.centru()
            rez.append(Triunghi(P.a, mab, mac))
            rez.append(Triunghi(mac, mab, mbc))
            rez.append(Triunghi(mab, mbc, P.b))
            rez.append(Triunghi(mac, mbc, P.c))
        return rez

    def traseaza(li):
        for k in range(len(li)):
            li[k].show(Color.Index(200 + k ))
            if C.mustClose(): return


    def liniaza(li):
        for k in range(1, len(li)):
            col = Color.Index(k // 5)
            C.drawLine(li[k - 1].centru(), li[k].centru(), col)
            if C.mustClose(): return

    C.setXminXmaxYminYmax(0, 10, 0, 10)
    C.fillScreen(Color.Navy)
    fig = [Triunghi(5*(0 + 0j),5*( 2+ 0j), 5*(1 + 2j))]
    # fig = [Patrat(0.5 + 1j, 1 + 9j, 7 + 8j, 9.5 + 1j)]
    nrEtape=5
    for k in range(nrEtape):
        fig = transforma(fig)
    traseaza(fig)
    # liniaza(fig)

def TriunghiLinie():
    class Triunghi:
        def __init__(self, a, b, c):
            self.a, self.b, self.c = a, b, c

        def show(self, col):
            C.fillNgon([self.a, self.b, self.c], col)
        def draw(self):
            C.drawLine(self.a,self.b,Color.Black)
            C.drawLine(self.b,self.c,Color.Black)
            C.drawLine(self.c,self.a,Color.Black)
            C.drawLine(self.a,self.centru(),Color.Black)
            C.drawLine(self.c,self.centru(),Color.Black)
        def centru(self):
            return (self.a + self.b + self.c ) / 3

    def transforma(li):
        rez = []
        for P in li:
            mab, mbc, mac = (P.a + P.b) / 2, (P.b + P.c) / 2, (P.c + P.a) / 2
            c0 = P.centru()
            rez.append(Triunghi(mab, mbc, P.b))
            rez.append(Triunghi(P.a, mab, mac))#nu
            rez.append(Triunghi(mab, mac, mbc))#nu
            rez.append(Triunghi(mac, P.c, mbc))#nu
        return rez

    def traseaza(li):
        for k in range(len(li)):
            li[k].show(Color.Index(200 + k ))
            if C.mustClose(): return


    def liniaza(li):
        for k in range(0, len(li)):
            col = Color.Index(k // 5)
            # print(li[k].a,li[k].centru())
            li[k].draw()
            # C.drawLine(li[k].centru(), li[k].c, col)
            # C.drawLine(li[k].b, li[k].centru(), col)
            if C.mustClose(): return

    C.setXminXmaxYminYmax(0, 10, 0, 10)
    C.fillScreen(Color.Whitesmoke)
    fig = [Triunghi(5*(0 + 0j),5*( 2+ 0j), 5*(1 + 2j))]
    # fig = [Patrat(0.5 + 1j, 1 + 9j, 7 + 8j, 9.5 + 1j)]
    nrEtape=1
    for k in range(nrEtape):
        fig = transforma(fig)
    print(len(fig))
    traseaza(fig)
    liniaza(fig)


####################################################################################TEMA 12

def Newton4prim1():
    eps0 = C.fromRhoTheta(1.0,  math.pi / 4.0)
    eps1 = C.fromRhoTheta(1.0, 3.0 * math.pi / 4.0)
    eps2 = C.fromRhoTheta(1.0, 5.0 * math.pi / 4.0)
    eps3= C.fromRhoTheta(1.0, 7.0*math.pi/4.0)
    def f(z):
        return z-(z**4+1)/(4*z**3) if z != 0 else 10e10

    c0 = 0
    r = 2
    C.setXminXmaxYminYmax(c0.real - r, c0.real + r, c0.imag - r, c0.imag + r)
    nrIter = 300
    for coloana in C.screenColumns():
        for zeta in coloana:
            col = Color.Black
            z = zeta
            for _ in range(nrIter):
                if abs(z - eps0) < 0.1:
                    col = Color.Darkblue
                    break
                if abs(z - eps1) < 0.1:
                    col = Color.Yellow
                    break
                if abs(z - eps2) < 0.1:
                    col = Color.Fuchsia
                    break
                if abs(z - eps3) < 0.1:
                    col = Color.Green
                    break
                z = f(z)
            C.setPixel(zeta, col)
        if C.mustClose(): return
    C.drawLine(c0 - r, c0 + r, Color.White)
    C.drawLine(c0 - r * 1j, c0 + r * 1j, Color.White)




def Newton3prim():
    eps0 = C.fromRhoTheta(1.0, 0.0 * math.pi / 3.0)
    eps1 = C.fromRhoTheta(1.0, 2.0 * math.pi / 3.0)
    eps2 = C.fromRhoTheta(1.0, 4.0 * math.pi / 3.0)

    def f(z):
        if z == 0:
            return 1.0e100
        else:
            return (2 * z * z * z + 1) / (3 * z * z)

    C.setXminXmaxYminYmax(-2, 2, -2, 2)
    nrIter = 300
    for coloana in C.screenColumns():
        for zeta in coloana:
            col = Color.Black
            z = zeta
            if z!=0:
                z=1/z
            for _ in range(nrIter):
                if C.rho(z - eps0) < 0.1:
                    col = Color.Darkblue
                    break
                if C.rho(z - eps1) < 0.1:
                    col = Color.Yellow
                    break
                if C.rho(z - eps2) < 0.1:
                    col = Color.Fuchsia
                    break
                z = f(z)
            C.setPixel(zeta, col)
        if C.mustClose(): return
    C.setAxis(Color.White)
    C.refreshScreen()

########################################################################################TEMA 13

def JuliaBazine():
    c = -0.21 - 0.7j

    def f(z):
        return z * z + c

    C.setXminXmaxYminYmax(-1.5, 1.5, -1.5, 1.5)
    C.fillScreen(Color.Black)
    nrIter = 1001 + 18
    rhoMax = 1.0e2
    for coloana in C.screenColumns():
        for zeta in coloana:
            z = zeta
            for k in range(nrIter):
                if abs(z) >= rhoMax: break
                z = f(z)
            if abs(z) < rhoMax:
                color = Color.Index(10 * sum(C.getHK(z)) + 200)
                C.setPixel(zeta, color)
                # C.setPixel(z,color)
        if C.mustClose():
            return


def Glyyn():
    c = -0.2
    x0 = 0.235
    y0 = 0.5
    l = 0.2

    def f(z):
        return z ** (1.5) + c

    C.setXminXmaxYminYmax(x0 - l, x0 + l, y0 - l, y0 + l)
    nrIter = 1000
    for coloana in C.screenColumns():
        for zeta in coloana:
            z = zeta
            for k in range(nrIter):
                if C.rho(z) > 4: break
                z = f(z)
            C.setPixel(zeta, Color.Index(100 + 5 * k))
        if C.mustClose():
            return



def JuliaPlina2():
    rhoMax = 1.0e2

    def f(z):
        if z == 0:
            return rhoMax
        u = z * z
        return u - 1 / u

    C.setXminXmaxYminYmax(-2, 2, -2, 2)
    nrIter = 100
    for coloana in C.screenColumns():
        for zeta in coloana:
            z = zeta
            for k in range(nrIter):
                z = f(z)
                if abs(z) > rhoMax: break
            C.setPixel(zeta, Color.Index(100 * k))
        if C.mustClose():
            break


def JuliaFinal():
    c = 0.001

    def f(z):
        if z == 0: return rhoMax
        return (z * z * z + c) / z

    C.setXminXmaxYminYmax(-1.1, 1.1, -1.1, 1.1)
    C.fillScreen(Color.Black)
    C.refreshScreen()
    rhoMax = 100
    nrIter = 107
    for coloana in C.screenColumns():
        for zeta in coloana:
            z = zeta
            for k in range(nrIter):
                if abs(z) > rhoMax: break
                z = f(z)
            if abs(z) <= rhoMax:
                col = Color.Index(abs(sum(C.getHK(20 * z))) + 650)
                C.setPixel(zeta, col)
                # C.setPixel(z, col)
        if C.mustClose():
            return

if __name__ == '__main__':
    C.initPygame()
    C.run(draw_triangle_colorat)
    C.run(triangle_circumcircle_regions)
    C.run(Mijloace)
    C.run(prob2)
    C.run(desenare)
    C.run(desenare2)
    C.run(Mediatoare)
    # C.run(LocGeomI)
    C.run(LocGeomH)
    C.run(HeptaPentagon2)
    C.run(HeptaPentagon3)
    C.run(GoldenRatio2024)
    C.run(PseudoSpiralaLuiArhimede)
    C.run(pentagon)
    C.run(SierpinskiPentagon)
    C.run(HilberTriunghi)
    C.run(TriunghiLinie)
    C.run(Newton3prim)
    C.run(Newton4prim1)
    C.run(Glyyn)
    C.run(JuliaFinal)
    C.run(JuliaPlina2)

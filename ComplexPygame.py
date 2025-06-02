# versiune: 01.06.2025
import pygame
import Color
import cmath

# variabile globale:
dim = 600
dimm = 599
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0
dxdh = 1.0 / dimm
dydk = 1.0 / dimm
dhdx = float(dimm)
dkdy = float(dimm)
screen = pygame.Surface((dim, dim))
mustExit = True
mustPainting = False

# redenumiri de functii:
rho = abs
theta = cmath.phase
fromRhoTheta = cmath.rect
wait = pygame.time.wait

def initPygame(dm=600):
    global screen, dim, dimm
    dim, dimm = dm, dm - 1
    print("start pygame")
    pygame.init()
    screen = pygame.display.set_mode((dim, dim))

def setXminXmaxYminYmax(xxmin=0.0, xxmax=1.0, yymin=0.0, yymax=1.0):
    global xmin, xmax, ymin, ymax, dxdh, dydk, dhdx, dkdy
    xmin = xxmin
    xmax = xxmax
    ymin = yymin
    ymax = yymax
    dxdh = (xmax - xmin) / dimm
    dydk = (ymax - ymin) / dimm
    dhdx = dimm / (xmax - xmin)
    dkdy = dimm / (ymax - ymin)

def getZ(h, k):
    # returneaza afixul z al punctului corespunzator pixelului (h,k) din ecran
    # (h, k) = (0, 0) este coltul din STANGA JOS
    return complex(xmin + h * dxdh, ymin + k * dydk)

def getXY(h, k):
    # returneaza punctul (x,y) corespunzator pixelului (h,k) din ecran
    return xmin + h * dxdh, ymin + k * dydk

def getHK(z):
    # returneaza pixelul  (h,k) corespunzator lui z complex
    return round((z.real - xmin) * dhdx), round((z.imag - ymin) * dkdy)

def _getIJ(z):
    # pentru uz privat: (i, j) == (h, dimm - k)
    # furnizeaza coordonatele (i,j) din bitmap
    # coordonate folosite de set_at si get_at
    # set_at((0,0),color) este coltul din STANGA SUS
    return round((z.real - xmin) * dhdx), round((ymax - z.imag) * dkdy)

def screenAffixes():
    # returneaza lista celor dim*dim numere complexe reprezentabile pe ecran
    return [getZ(h, k) for h in range(dim) for k in range(dim)]

def screenColumns():
    # genereaza dim liste cu numerele complexe reprezentabile
    # asezate pe coloane
    for h in range(dim):
        yield [getZ(h, k) for k in range(dim)]

def setPixelHK(h, k, color):
    # seteaza pe ecran pixelul de coordonate (h,k)
    screen.set_at((h, dimm - k), color)

def setPixelXY(x, y, color):
    # seteaza pixelul corespunzator punctului  (x,y) din plan
    screen.set_at((round((x - xmin) * dhdx), round((ymax - y) * dkdy)), color)

def setPixel(z, color):
    # seteaza pixelul corespunzator numarului complex z
    screen.set_at((round((z.real - xmin) * dhdx), round((ymax - z.imag) * dkdy)), color)

def drawLineHK(h0, k0, h1, k1, color):
    # traseaza segmentul (h0,k0) (h1,k1) pe ecran
    pygame.draw.line(screen, color, (h0, dimm - k0), (h1, dimm - k1))

def drawLineXY(x0, y0, x1, y1, color):
    # traseaza segmentul (x0,y0) (x1,y1) din plan
    pygame.draw.line(screen, color, (round((x0 - xmin) * dhdx), round((ymax - y0) * dkdy)),
                     (round((x1 - xmin) * dhdx), round((ymax - y1) * dkdy)))

def drawLine(z0, z1, color):
    # traseaza segmentul z0 z1 din planul complex
    pygame.draw.line(screen, color, (round((z0.real - xmin) * dhdx), round((ymax - z0.imag) * dkdy)),
                     (round((z1.real - xmin) * dhdx), round((ymax - z1.imag) * dkdy)))

def setNgon(z_list, color):
    # seteaza varfurile poligonului z_list = [z0, z1, ...]
    for z in z_list:
        setPixel(z, color)
    return

def drawNgon(z_list, color):
    # traseaza laturile poligonului z_list = [z0, z1, ...]
    pygame.draw.polygon(screen, color, [_getIJ(z) for z in z_list], width=1)

def fillNgon(z_list, color):
    # umple poligonul convex z_list = [z0, z1, ...]
    pygame.draw.polygon(screen, color, [_getIJ(z) for z in z_list])

def setAxis(color=Color.Black):
    drawLineXY(xmin, 0, xmax, 0, color)
    drawLineXY(0, ymin, 0, ymax, color)

def setTextIJ(text=" ", i=dim // 2, j=dim // 2, color=Color.Navy, size=16):
    # scrie un text pe ecran, (i, j) = (0, 0) este coltul de STANGA SUS
    myFont = pygame.font.Font('TimeRomanNormal.ttf', size)
    textImag = myFont.render(text, True, color)
    textRect = textImag.get_rect()
    textRect.bottomleft = (i, j)
    screen.blit(textImag, textRect)

def setText(text="O", z=0j, color=Color.Black, size=16):
    # scrie un text pe ecran in dreptul lui z
    myFont = pygame.font.Font('TimeRomanNormal.ttf', size)
    textImag = myFont.render(text, True, color)
    textRect = textImag.get_rect()
    i, j = _getIJ(z)
    textRect.center = (i, j - 3 * size // 4)
    screen.blit(textImag, textRect)

def saveScreenPNG(filename):
    pygame.image.save(screen, filename + ".png")

def fillScreen(color=Color.Whitesmoke):
    screen.fill(color)

def refreshScreen():
    pygame.display.flip()

def run(drawing_function):
    # lanseaza in executie functia care realizeaza desenul
    print(f"start run({drawing_function.__name__})")
    fillScreen()
    pygame.display.set_caption(drawing_function.__name__ + " : apasati bara de spatiu ")
    pygame.display.flip()
    global mustExit, mustPainting
    while True:  # main loop
        mustExit = False
        mustPainting = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                mustExit = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                mustPainting = True
        if mustExit:
            break  # main loop
        if mustPainting:
            pygame.display.set_caption(drawing_function.__name__ + " : in lucru")
            # lansam desenarea:
            drawing_function()
            # dupa terminarea desenarii:
            pygame.display.flip()
            if mustExit:
                break  # main loop
            if mustPainting:
                pygame.display.set_caption(drawing_function.__name__ + " : complet")
            else:
                pygame.display.set_caption(drawing_function.__name__ + " : oprit")
    print(f"exit run({drawing_function.__name__})")

def mustClose():
    # poate fi apelata in timpul desenarii, pentru oprirea desenarii
    pygame.display.flip()
    global mustExit, mustPainting
    mustStop = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            mustExit = True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            mustStop = True
            mustPainting = False
    return mustExit or mustStop

if __name__ == '__main__':
    # exemplu de utilizare:
    def desenare():
        setXminXmaxYminYmax(-1, 1, -1, 1)
        fillScreen(Color.Darkturquoise)
        setTextIJ("Pentru oprire apasati bara de spatiu", 10, 20)
        for k in range(10 ** 6):
            z = fromRhoTheta(0.8, k * 0.0005)
            v = fromRhoTheta(0.2, k * 0.009)
            drawLine(z, v, Color.Index(k // 10))
            if mustClose():
                break
        fillScreen(Color.Turquoise)
        setTextIJ("Pentru repornire apasati tot bara de spatiu", 10, 20)
        refreshScreen()

    initPygame()
    run(desenare)

# Contagens não estão precisas
import cv2
import numpy as np

# Vídeo
video = cv2.VideoCapture("cars-video.mp4")

# Detector de movimento
detector = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=40,
    detectShadows=False
)

contador = 0

# Linha de contagem
linha_y = 600
offset = 20

# Centros já contados
centros_contados = []

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # Detecta movimento
    mascara = detector.apply(frame)

    kernel = np.ones((5, 5), np.uint8)

    mascara = cv2.erode(mascara, kernel, iterations=1)
    mascara = cv2.dilate(mascara, kernel, iterations=4)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Desenha linha de contagem
    cv2.line(
        frame,
        (0, linha_y),
        (1280, linha_y),
        (0, 255, 255),
        3
    )

    for cnt in contornos:

        area = cv2.contourArea(cnt)

        # Ignora objetos pequenos
        if area < 3500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        cx = int(x + w / 2)
        cy = int(y + h / 2)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )

        # Verifica se cruzou a linha
        if linha_y - offset < cy < linha_y + offset:

            novo_carro = True

            for px, py in centros_contados:

                distancia = np.sqrt(
                    (cx - px) ** 2 +
                    (cy - py) ** 2
                )

                # Mesmo carro já contado
                if distancia < 80:
                    novo_carro = False
                    break

            if novo_carro:

                contador += 1
                centros_contados.append((cx, cy))

                print(f"Carro contado! Total: {contador}")

                cv2.line(
                    frame,
                    (0, linha_y),
                    (1280, linha_y),
                    (0, 255, 0),
                    5
                )

    # Limpa centros muito antigos
    if len(centros_contados) > 200:
        centros_contados = centros_contados[-100:]

    # Exibe contador
    cv2.putText(
        frame,
        f"CARROS: {contador}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 0, 0),
        3
    )

    cv2.imshow("Contador de Carros", frame)
    cv2.imshow("Mascara", mascara)

    tecla = cv2.waitKey(30)

    if tecla == 27:  # ESC
        break

video.release()
cv2.destroyAllWindows()
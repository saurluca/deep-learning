## Aufgabe 1
Wie nennt man die geometrische Figur, die die Punkte im Raum trennt, die von einem
k ̈unstlichen Neuron unterschiedlich klassifiziert werden, wenn der Raum mehr als zwei Di-
mensionen hat?
 

 Eine Hyperbene

## Aufgabe 2

Eckpunkte des Trapezoids:
P(-4, -2), K(-2, 2), R(2, 4), Q(8, 2)

Mathematischer Ansatz:
1. Für jede Seite des Trapezoids leiten wir eine Lineare Gleichung in der Form ax + by + c = 0 ab
2. Wir stellen sicher, dass für Punkte innerhalb des Trapezoids, ax + by + c > 0
3. Die Neuronen der verborgenen Schicht berechnen diese Linearen Gleichungen
4. Der minimale Wert über alle vier Linien bestimmt, ob ein Punkt innerhalb (min > 0) oder außerhalb (min ≤ 0) liegt

- Gewichte (W):
     [ [4, -2],    # Neuron h₁ (Linie PK)
       [2, -4],    # Neuron h₂ (Linie KR)
       [-2, -6],   # Neuron h₃ (Linie RQ)
       [-4, 12] ]  # Neuron h₄ (Linie QP)

- Biases (b):
    [12, 12, 28, 8]

## Aufgabe C

### Klassifikation von MNIST Bildern

**Klasse von Aufgaben T:** Die Zuordnung von handgeschriebenen Ziffern (MNIST Bilder, 28×28 Pixel) zu einer der 10 Ziffernklassen (0-9).

**Performanz-Maß P:** Die Kreuzentropie-Verlustfunktion für Mehrklassenklassifikation:
L = -∑(y_i * log(ŷ_i))
Dabei ist y_i die tatsächliche Klasse (als One-Hot-Vektor) und ŷ_i die vorhergesagte Wahrscheinlichkeit für jede Klasse.

**Erfahrung E:** Der MNIST-Trainingsdatensatz mit 60.000 Bildern von handgeschriebenen Ziffern. Jedes Bild ist 28×28 Pixel groß mit Grauwerten zwischen 0-255 und hat ein Label zwischen 0-9. Das neuronale Netz lernt, indem es wiederholt Vorhersagen macht, den Fehler berechnet und die Gewichte durch Backpropagation anpasst.

### Regression von Surface Reflectance Parametern

**Klasse von Aufgaben T:** Die Schätzung von Oberflächeneigenschaften (wie Rauheit, Glanz, Farbe) aus 2D-Bildern von 3D-Objekten.

**Performanz-Maß P:** Der mittlere quadratische Fehler (MSE):
L = 1/n * ∑(y_i - ŷ_i)²
Wobei y_i die tatsächlichen Reflexionsparameter und ŷ_i die vorhergesagten Parameter sind.

**Erfahrung E:** Ein Datensatz aus Bildern von 3D-Objekten mit bekannten Oberflächeneigenschaften. Für jedes Bild sind die korrekten Reflexionsparameter (kontinuierliche Werte) verfügbar. Das Netzwerk lernt die Abbildung von Bildmerkmalen auf diese Parameter durch wiederholtes Training, Fehlerberechnung und Gewichtsanpassung mittels Gradientenabstieg.

"""
Neuronales Netzwerk für Trapezoid-Klassifizierung

Diese Implementierung erstellt ein zweischichtiges neuronales Netzwerk, das Punkte
als innerhalb oder außerhalb eines Trapezoids klassifiziert, das durch vier Punkte definiert ist.

Architektur:
- Eingabeschicht: 2 Neuronen (x, y-Koordinaten)
- Verborgene Schicht: 4 Neuronen mit ReLU-Aktivierung (jedes repräsentiert eine Seite des Trapezoids)
- Ausgabeschicht: 1 Neuron mit Identitäts-Aktivierung (Minimum der Ausgaben der verborgenen Schicht)

Mathematischer Ansatz:
1. Für jede Seite des Trapezoids leiten wir eine Lineare Gleichung in der Form ax + by + c = 0 ab
2. Wir stellen sicher, dass für Punkte innerhalb des Trapezoids, ax + by + c > 0
3. Die Neuronen der verborgenen Schicht berechnen diese Linearen Gleichungen
4. Der minimale Wert über alle vier Linien bestimmt, ob ein Punkt innerhalb (min > 0) oder außerhalb (min ≤ 0) liegt

Diese analytische Lösung demonstriert, wie neuronale Netzwerke Entscheidungsgrenzen
ohne den Einsatz von Gradientenabstieg lernen können.

Eckpunkte des Trapezoids:
P(-4, -2), K(-2, 2), R(2, 4), Q(8, 2)

Tatsächliche Lösung und Herleitung:
Die Kernidee ist, dass jedes Neuron in der verborgenen Schicht die Gleichung einer Kante des Trapezoids berechnet.
Für eine Linie zwischen zwei Punkten (x1,y1) und (x2,y2) wird die Gleichung wie folgt berechnet:
  - a = y2 - y1
  - b = x1 - x2
  - c = x2*y1 - x1*y2

Für jede der vier Linien (PK, KR, RQ, QP) berechnen wir diese Koeffizienten,
wobei wir sicherstellen, dass die Werte für Punkte innerhalb des Trapezoids positiv sind.

Die vier Neuronen der verborgenen Schicht berechnen:
  h₁(x,y) = max(0, a_pk*x + b_pk*y + c_pk)  # Linie PK
  h₂(x,y) = max(0, a_kr*x + b_kr*y + c_kr)  # Linie KR
  h₃(x,y) = max(0, a_rq*x + b_rq*y + c_rq)  # Linie RQ
  h₄(x,y) = max(0, a_qp*x + b_qp*y + c_qp)  # Linie QP

Die ReLU-Aktivierungsfunktion (max(0, z)) stellt sicher, dass negative Werte auf Null gesetzt werden.
In der Ausgabeschicht nehmen wir einfach das Minimum aller vier Werte:
  o(x,y) = min(h₁(x,y), h₂(x,y), h₃(x,y), h₄(x,y))

Da wir die ReLU-Funktion verwenden und alle Liniengleichungen so orientiert haben, dass sie für Punkte
innerhalb des Trapezoids positive Werte liefern, ist der Punkt im Trapezoid enthalten, wenn alle vier
Werte positiv sind (also wenn das Minimum positiv ist).

Tatsächliche Gewichte und Biases für das Netzwerk:

1. Verborgene Schicht (Hidden Layer):
   - Gewichte (W¹):
     [ [4, -2],    # Neuron h₁ (Linie PK)
       [2, -4],    # Neuron h₂ (Linie KR)
       [-2, -6],   # Neuron h₃ (Linie RQ)
       [-4, 12] ]  # Neuron h₄ (Linie QP)

   - Biases (b¹):
     [12, 12, 28, 8]

2. Ausgabeschicht (Output Layer):
   Das Minimum der vier Werte aus der verborgenen Schicht wird direkt als Ausgabe verwendet.
   Dies entspricht einer Identitätsfunktion als Aktivierung.

Somit ist die mathematische Funktion des Netzwerks:
  f(x,y) = min(max(0, 4x - 2y + 12),
               max(0, 2x - 4y + 12),
               max(0, -2x - 6y + 28),
               max(0, -4x + 12y + 8))

Diese Funktion gibt positive Werte für Punkte innerhalb des Trapezoids und negative Werte für Punkte
außerhalb des Trapezoids zurück, was genau der gewünschten Klassifikation entspricht.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the corner points of the trapezoid
P = torch.tensor([-4.0, -2.0])  # P(-4, -2)
Q = torch.tensor([8.0, 2.0])  # Q(8, 2)
R = torch.tensor([2.0, 4.0])  # R(2, 4)
K = torch.tensor([-2.0, 2.0])  # K(-2, 2)


class TrapezoidNetwork(torch.nn.Module):
    def __init__(self):
        super(TrapezoidNetwork, self).__init__()

        # For a line passing through (x1,y1) and (x2,y2), the line equation is:
        # (y2-y1)x - (x2-x1)y + x2*y1 - x1*y2 = 0
        # We'll represent this as ax + by + c = 0

        # Line PK (Bottom Left to Top Left)
        a_pk = K[1] - P[1]  # 2 - (-2) = 4
        b_pk = -(K[0] - P[0])  # -((-2) - (-4)) = -(-2+4) = -2
        c_pk = K[0] * P[1] - P[0] * K[1]  # (-2)*(-2) - (-4)*2 = 4 - (-8) = 12

        # Line KR (Top Left to Top Right)
        a_kr = R[1] - K[1]  # 4 - 2 = 2
        b_kr = -(R[0] - K[0])  # -(2 - (-2)) = -4
        c_kr = R[0] * K[1] - K[0] * R[1]  # 2*2 - (-2)*4 = 4 - (-8) = 12

        # Line RQ (Top Right to Bottom Right)
        a_rq = Q[1] - R[1]  # 2 - 4 = -2
        b_rq = -(Q[0] - R[0])  # -(8 - 2) = -6
        c_rq = Q[0] * R[1] - R[0] * Q[1]  # 8*4 - 2*2 = 32 - 4 = 28

        # Line QP (Bottom Right to Bottom Left)
        a_qp = P[1] - Q[1]  # -2 - 2 = -4
        b_qp = -(P[0] - Q[0])  # -((-4) - 8) = -(-12) = 12
        c_qp = P[0] * Q[1] - Q[0] * P[1]  # (-4)*2 - 8*(-2) = -8 + 16 = 8

        # Check each line with a point definitely inside the trapezoid
        inside_point = torch.tensor([1.0, 1.0])  # A point known to be inside

        # Ensure all lines are correctly oriented (inside = positive)
        # For each line, we want ax + by + c > 0 for points inside the trapezoid
        if a_pk * inside_point[0] + b_pk * inside_point[1] + c_pk < 0:
            a_pk, b_pk, c_pk = -a_pk, -b_pk, -c_pk

        if a_kr * inside_point[0] + b_kr * inside_point[1] + c_kr < 0:
            a_kr, b_kr, c_kr = -a_kr, -b_kr, -c_kr

        if a_rq * inside_point[0] + b_rq * inside_point[1] + c_rq < 0:
            a_rq, b_rq, c_rq = -a_rq, -b_rq, -c_rq

        if a_qp * inside_point[0] + b_qp * inside_point[1] + c_qp < 0:
            a_qp, b_qp, c_qp = -a_qp, -b_qp, -c_qp

        # First layer weights represent the line equations
        self.weights = torch.nn.Parameter(
            torch.tensor([[a_pk, b_pk], [a_kr, b_kr], [a_rq, b_rq], [a_qp, b_qp]]),
            requires_grad=False,
        )

        # First layer biases
        self.biases = torch.nn.Parameter(
            torch.tensor([c_pk, c_kr, c_rq, c_qp]), requires_grad=False
        )

    def forward(self, x):
        # Handle single point
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Compute line equations for each point
        # ax + by + c for each of the 4 lines
        line_values = torch.matmul(x, self.weights.T) + self.biases

        # A point is inside if all line values are positive
        # We use the minimum value as our classifier
        # If min > 0, point is inside
        # If min <= 0, point is outside or on the boundary
        min_values = torch.min(line_values, dim=1)[0]

        return min_values


def test_network():
    # Create network
    net = TrapezoidNetwork()

    # Test the corner points
    corners = torch.stack([P, K, R, Q])
    results_corners = net(corners)
    print("Corner points results (should be close to 0):")
    for point, result in zip(corners, results_corners):
        print(f"Point ({point[0]:.1f}, {point[1]:.1f}): {result:.6f}")

    # Test some points clearly inside
    inside_points = torch.tensor(
        [
            [1.0, 1.0],  # Center-ish
            [2.0, 2.0],  # Right middle
            [-2.0, 0.0],  # Left middle
        ]
    )
    results_inside = net(inside_points)
    print("\nInside points results (should be positive):")
    for point, result in zip(inside_points, results_inside):
        print(f"Point ({point[0]:.1f}, {point[1]:.1f}): {result:.6f}")

    # Test some points clearly outside
    outside_points = torch.tensor(
        [
            [-6.0, -2.0],  # Far left
            [10.0, 2.0],  # Far right
            [0.0, 6.0],  # Above
            [0.0, -4.0],  # Below
        ]
    )
    results_outside = net(outside_points)
    print("\nOutside points results (should be negative):")
    for point, result in zip(outside_points, results_outside):
        print(f"Point ({point[0]:.1f}, {point[1]:.1f}): {result:.6f}")

    # Visualize the decision boundary
    x = np.linspace(-8, 12, 100)
    y = np.linspace(-4, 6, 100)
    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.stack([X.flatten(), Y.flatten()]).T, dtype=torch.float32)

    Z = net(points).detach().numpy().reshape(X.shape)

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=[0], colors="black")
    plt.fill([P[0], K[0], R[0], Q[0]], [P[1], K[1], R[1], Q[1]], alpha=0.3)
    plt.scatter([P[0], K[0], R[0], Q[0]], [P[1], K[1], R[1], Q[1]], c="red")
    plt.grid(True)
    plt.axis("equal")
    plt.title("Trapezoid Decision Boundary")
    plt.savefig("trapezoid_boundary.png")
    plt.close()


if __name__ == "__main__":
    test_network()

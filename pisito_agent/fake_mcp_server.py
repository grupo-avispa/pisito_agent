from fastmcp import FastMCP
import os
import json
import requests

mcp = FastMCP('Fake-MCP-server')
        
@mcp.tool(
    name='light_control',
    description='''Controla las luces de las habitaciones.
Input: room (string) - Nombre de la sala ("salón", "cocina", "dormitorio", "salita", "habitación invitados")
    action (string) - Acción a realizar ("ON", "OFF")
Output: status (string) - Confirmación de la acción realizada''')
async def light_control(room: str, action: str):
    print(f"Light control - Room: {room}, Action: {action}")
    valid_rooms = ["salón", "cocina", "dormitorio", "salita", "habitación invitados"]
    valid_actions = ["ON", "OFF"]
    
    if room not in valid_rooms:
        return f"Error: Habitación '{room}' no válida. Habitaciones disponibles: {', '.join(valid_rooms)}"
    if action not in valid_actions:
        return f"Error: Acción '{action}' no válida. Acciones disponibles: ON, OFF"
    
    return f"Luz {'encendida' if action == 'ON' else 'apagada'} en {room}"

@mcp.tool(
    name='temperature_control',
    description='''Controla la temperatura del termostato o aire acondicionado.
Input: device_name (string) - Tipo de dispositivo ("termostato", "aire acondicionado")
    room (string) - Sala donde se encuentra el dispositivo ("salón", "cocina", "dormitorio", "salita", "habitación invitados")
    temperature (float) - Temperatura objetivo en grados Celsius
Output: status (string) - Confirmación del ajuste de temperatura''')
async def temperature_control(device_name: str, room: str, temperature: float):
    print(f"Temperature control - Device: {device_name}, Room: {room}, Temp: {temperature}")
    valid_devices = ["termostato", "aire acondicionado"]
    valid_rooms = ["salón", "cocina", "dormitorio", "salita", "habitación invitados"]
    
    if device_name not in valid_devices:
        return f"Error: Dispositivo '{device_name}' no válido. Dispositivos disponibles: {', '.join(valid_devices)}"
    if room not in valid_rooms:
        return f"Error: Habitación '{room}' no válida. Habitaciones disponibles: {', '.join(valid_rooms)}"
    if temperature < 16 or temperature > 30:
        return f"Error: Temperatura {temperature}°C fuera de rango (16-30°C)"
    
    return f"{device_name.capitalize()} ajustado a {temperature}°C en {room}"

@mcp.tool(
    name='blind_control',
    description='''Controla la posición de las persianas en una habitación.
Input: room (string) - Sala donde se controlarán las persianas ("salón", "cocina", "dormitorio", "salita", "habitación invitados")
    action (string) - Acción a realizar ("subir", "bajar", "parar")
Output: status (string) - Confirmación de la acción realizada''')
async def blind_control(room: str, action: str):
    print(f"Blind control - Room: {room}, Action: {action}")
    valid_rooms = ["salón", "cocina", "dormitorio", "salita", "habitación invitados"]
    valid_actions = ["subir", "bajar", "parar"]
    
    if room not in valid_rooms:
        return f"Error: Habitación '{room}' no válida. Habitaciones disponibles: {', '.join(valid_rooms)}"
    if action not in valid_actions:
        return f"Error: Acción '{action}' no válida. Acciones disponibles: {', '.join(valid_actions)}"
    
    action_text = {"subir": "subiendo", "bajar": "bajando", "parar": "detenidas"}
    return f"Persianas {action_text[action]} en {room}"

@mcp.tool(
    name='door_lock_control',
    description='''Controla el estado de las cerraduras inteligentes.
Input: door (string) - Puerta que se desea controlar ("puerta principal", "puerta trasera", "garaje")
    action (string) - Acción a realizar ("bloquear", "desbloquear")
Output: status (string) - Confirmación del estado de la cerradura''')
async def door_lock_control(door: str, action: str):
    print(f"Door lock control - Door: {door}, Action: {action}")
    valid_doors = ["puerta principal", "puerta trasera", "garaje"]
    valid_actions = ["bloquear", "desbloquear"]
    
    if door not in valid_doors:
        return f"Error: Puerta '{door}' no válida. Puertas disponibles: {', '.join(valid_doors)}"
    if action not in valid_actions:
        return f"Error: Acción '{action}' no válida. Acciones disponibles: bloquear, desbloquear"
    
    action_text = {"bloquear": "bloqueada", "desbloquear": "desbloqueada"}
    return f"{door.capitalize()} {action_text[action]} correctamente"

@mcp.tool(
    name='music_control',
    description='''Controla la reproducción de música en distintas habitaciones.
Input: room (string) - Sala donde se reproducirá la música ("salón", "cocina", "dormitorio", "salita", "habitación invitados")
    action (string) - Acción sobre la música ("play", "pause", "stop")
    volume (int) - Volumen de reproducción (0 a 100) [opcional]
Output: status (string) - Confirmación de la acción de reproducción''')
async def music_control(room: str, action: str, volume: int = None):
    print(f"Music control - Room: {room}, Action: {action}, Volume: {volume}")
    valid_rooms = ["salón", "cocina", "dormitorio", "salita", "habitación invitados"]
    valid_actions = ["play", "pause", "stop"]
    
    if room not in valid_rooms:
        return f"Error: Habitación '{room}' no válida. Habitaciones disponibles: {', '.join(valid_rooms)}"
    if action not in valid_actions:
        return f"Error: Acción '{action}' no válida. Acciones disponibles: play, pause, stop"
    if volume is not None and (volume < 0 or volume > 100):
        return f"Error: Volumen {volume} fuera de rango (0-100)"
    
    action_text = {"play": "reproduciendo", "pause": "pausada", "stop": "detenida"}
    volume_text = f" con volumen {volume}" if volume is not None else ""
    return f"Música {action_text[action]} en {room}{volume_text}"

@mcp.tool(
    name='presence_detection',
    description='''Activa o desactiva los sensores de movimiento o presencia en las habitaciones.
Input: room (string) - Sala donde se ajustará el sensor ("salón", "cocina", "dormitorio", "salita", "habitación invitados")
    action (string) - Acción sobre el sensor ("activar", "desactivar")
Output: status (string) - Confirmación del estado del sensor''')
async def presence_detection(room: str, action: str):
    print(f"Presence detection - Room: {room}, Action: {action}")
    valid_rooms = ["salón", "cocina", "dormitorio", "salita", "habitación invitados"]
    valid_actions = ["activar", "desactivar"]
    
    if room not in valid_rooms:
        return f"Error: Habitación '{room}' no válida. Habitaciones disponibles: {', '.join(valid_rooms)}"
    if action not in valid_actions:
        return f"Error: Acción '{action}' no válida. Acciones disponibles: activar, desactivar"
    
    action_text = {"activar": "activado", "desactivar": "desactivado"}
    return f"Sensor de presencia {action_text[action]} en {room}"


if __name__ == '__main__':
    mcp.run(transport='http', host='127.0.0.1', port=3002)
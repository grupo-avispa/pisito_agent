from fastmcp import FastMCP
import os
import json
import requests

mcp = FastMCP('Fake-MCP-server')
        
@mcp.tool(
    name='light_control',
    description='''Controls room lights.
Input: room (string) - Room name ("living room", "kitchen", "bedroom", "study", "guest room")
action (string) - Action to perform ("ON", "OFF")
Output: status (string) - Action confirmation''')
async def light_control(room: str, action: str):
    print(f"Light control - Room: {room}, Action: {action}")
    valid_rooms = ["living room", "kitchen", "bedroom", "study", "guest room"]
    valid_actions = ["ON", "OFF"]
    
    if room not in valid_rooms:
        return f"Error: Invalid room '{room}'. Available rooms: {', '.join(valid_rooms)}"
    if action not in valid_actions:
        return f"Error: Invalid action '{action}'. Available actions: ON, OFF"
    
    return f"Light turned {'on' if action == 'ON' else 'off'} in {room}"

@mcp.tool(
    name='temperature_control',
    description='''Controls thermostat or air conditioning temperature.
Input: device_name (string) - Device type ("thermostat", "air conditioner")
room (string) - Room location ("living room", "kitchen", "bedroom", "study", "guest room")
temperature (float) - Target temperature in Celsius
Output: status (string) - Temperature adjustment confirmation''')
async def temperature_control(device_name: str, room: str, temperature: float):
    print(f"Temperature control - Device: {device_name}, Room: {room}, Temp: {temperature}")
    valid_devices = ["thermostat", "air conditioner"]
    valid_rooms = ["living room", "kitchen", "bedroom", "study", "guest room"]
    
    if device_name not in valid_devices:
        return f"Error: Invalid device '{device_name}'. Available devices: {', '.join(valid_devices)}"
    if room not in valid_rooms:
        return f"Error: Invalid room '{room}'. Available rooms: {', '.join(valid_rooms)}"
    if temperature < 16 or temperature > 30:
        return f"Error: Temperature {temperature}°C out of range (16-30°C)"
    
    return f"{device_name.capitalize()} set to {temperature}°C in {room}"

@mcp.tool(
    name='blind_control',
    description='''Controls window blinds position.
Input: room (string) - Room location ("living room", "kitchen", "bedroom", "study", "guest room")
action (string) - Action to perform ("raise", "lower", "stop")
Output: status (string) - Action confirmation''')
async def blind_control(room: str, action: str):
    print(f"Blind control - Room: {room}, Action: {action}")
    valid_rooms = ["living room", "kitchen", "bedroom", "study", "guest room"]
    valid_actions = ["raise", "lower", "stop"]
    
    if room not in valid_rooms:
        return f"Error: Invalid room '{room}'. Available rooms: {', '.join(valid_rooms)}"
    if action not in valid_actions:
        return f"Error: Invalid action '{action}'. Available actions: {', '.join(valid_actions)}"
    
    action_text = {"raise": "raising", "lower": "lowering", "stop": "stopped"}
    return f"Blinds {action_text[action]} in {room}"

@mcp.tool(
    name='door_lock_control',
    description='''Controls smart door locks.
Input: door (string) - Door to control ("front door", "back door", "garage")
action (string) - Action to perform ("lock", "unlock")
Output: status (string) - Lock state confirmation''')
async def door_lock_control(door: str, action: str):
    print(f"Door lock control - Door: {door}, Action: {action}")
    valid_doors = ["front door", "back door", "garage"]
    valid_actions = ["lock", "unlock"]
    
    if door not in valid_doors:
        return f"Error: Invalid door '{door}'. Available doors: {', '.join(valid_doors)}"
    if action not in valid_actions:
        return f"Error: Invalid action '{action}'. Available actions: lock, unlock"
    
    action_text = {"lock": "locked", "unlock": "unlocked"}
    return f"{door.capitalize()} {action_text[action]} successfully"

@mcp.tool(
    name='music_control',
    description='''Controls music playback in rooms.
Input: room (string) - Room location ("living room", "kitchen", "bedroom", "study", "guest room")
action (string) - Playback action ("play", "pause", "stop")
volume (int) - Playback volume (0 to 100) [optional]
Output: status (string) - Playback action confirmation''')
async def music_control(room: str, action: str, volume: int = None):
    print(f"Music control - Room: {room}, Action: {action}, Volume: {volume}")
    valid_rooms = ["living room", "kitchen", "bedroom", "study", "guest room"]
    valid_actions = ["play", "pause", "stop"]
    
    if room not in valid_rooms:
        return f"Error: Invalid room '{room}'. Available rooms: {', '.join(valid_rooms)}"
    if action not in valid_actions:
        return f"Error: Invalid action '{action}'. Available actions: play, pause, stop"
    if volume is not None and (volume < 0 or volume > 100):
        return f"Error: Volume {volume} out of range (0-100)"
    
    action_text = {"play": "playing", "pause": "paused", "stop": "stopped"}
    volume_text = f" at volume {volume}" if volume is not None else ""
    return f"Music {action_text[action]} in {room}{volume_text}"

@mcp.tool(
    name='presence_detection',
    description='''Controls motion/presence sensors.
Input: room (string) - Room location ("living room", "kitchen", "bedroom", "study", "guest room")
action (string) - Sensor action ("enable", "disable")
Output: status (string) - Sensor state confirmation''')
async def presence_detection(room: str, action: str):
    print(f"Presence detection - Room: {room}, Action: {action}")
    valid_rooms = ["living room", "kitchen", "bedroom", "study", "guest room"]
    valid_actions = ["enable", "disable"]
    
    if room not in valid_rooms:
        return f"Error: Invalid room '{room}'. Available rooms: {', '.join(valid_rooms)}"
    if action not in valid_actions:
        return f"Error: Invalid action '{action}'. Available actions: enable, disable"
    
    action_text = {"enable": "enabled", "disable": "disabled"}
    return f"Presence sensor {action_text[action]} in {room}"


if __name__ == '__main__':
    mcp.run(transport='http', host='127.0.0.1', port=3002)
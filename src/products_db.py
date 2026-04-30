"""Product catalog backed by a local SQLite database.

The agent reads from this catalog via function tools. Read-only by design —
no method here mutates data once seeded, so the LLM cannot alter inventory.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SEED_PRODUCTS: list[tuple[str, int, str, int]] = [
    ("Wireless Headphones", 1290, "Over-ear bluetooth headphones with 30h battery.", 1),
    ("Smart Watch", 3990, "Fitness tracker with heart-rate and sleep monitoring.", 1),
    ("Bluetooth Speaker", 1590, "Portable waterproof speaker, 12h playtime.", 1),
    ("USB-C Cable", 290, "1.5m braided USB-C to USB-C cable, 100W rated.", 1),
    ("Power Bank 10000mAh", 690, "Slim power bank with USB-C and USB-A outputs.", 1),
    ("Laptop Stand", 850, "Adjustable aluminum laptop stand, fits up to 17 inch.", 1),
    ("Wireless Mouse", 590, "Silent click wireless mouse with USB receiver.", 0),
    ("Mechanical Keyboard", 2490, "75% layout mechanical keyboard, hot-swappable.", 1),
    ("External SSD 1TB", 3290, "High-speed portable solid state drive, USB 3.2 gen 2.", 1),
    ("Gaming Mouse Pad", 450, "Large cloth surface with anti-slip rubber base.", 1),
    ("Webcam 1080p", 1890, "Full HD webcam with built-in microphone and privacy shutter.", 1),
    ("HDMI Cable 2.1", 490, "2m ultra high speed HDMI cable, supports 4K@120Hz.", 1),
    ("Wireless Charger Pad", 790, "15W fast wireless charging pad for Qi-enabled devices.", 1),
    ("Noise Cancelling Buds", 5490, "Premium true wireless earbuds with active noise cancellation.", 1),
    ("Smartphone Tripod", 350, "Flexible mini tripod with universal phone mount.", 1),
    ("Laptop Sleeve 14-inch", 590, "Water-resistant protective sleeve with soft lining.", 1),
    ("USB-C Hub 7-in-1", 1290, "Type-C adapter with HDMI, USB 3.0, and SD card slots.", 1),
    ("Portable Bluetooth Keyboard", 890, "Ultra-slim foldable keyboard for tablets and phones.", 1),
    ("Smart LED Bulb", 490, "WiFi-connected RGB bulb, compatible with Alexa and Google Home.", 1),
    ("Mini DLP Projector", 8990, "Portable pocket projector with built-in battery and speakers.", 0),
    ("Desk Lamp", 1190, "LED desk lamp with adjustable brightness and color temperature.", 1),
    ("Vertical Ergonomic Mouse", 1250, "Wireless vertical mouse designed to reduce wrist strain.", 1),
    ("Monitor Arm Dual", 1890, "Heavy-duty dual monitor mount for 13-27 inch screens.", 1),
    ("USB Microphone", 2190, "Condenser microphone for streaming and podcasting.", 1),
    ("Ring Light 10-inch", 650, "Dimmable LED ring light with tripod stand for video calls.", 1),
    ("Ethernet Cable Cat6 5m", 190, "High-speed network cable for stable wired connections.", 1),
    ("SD Card 128GB", 790, "UHS-I speed class 10 microSDXC card with adapter.", 1),
    ("Cable Management Box", 390, "Large cable organizer box to hide messy wires.", 1),
    ("Graphic Tablet", 2890, "Digital drawing tablet with battery-free stylus.", 1),
    ("Bluetooth Audio Receiver", 550, "Portable adapter to add bluetooth to wired speakers.", 1),
    ("Gaming Headset", 1690, "Wired gaming headset with 7.1 surround sound and mic.", 1),
    ("Laptop Backpack", 1490, "Anti-theft travel backpack with USB charging port.", 1),
    ("Wireless Presenter", 450, "PowerPoint remote with red laser pointer.", 1),
    ("Air Duster Can", 220, "Compressed air for cleaning electronics and keyboards.", 1),
    ("Smart Plug WiFi", 390, "Remote control power socket with energy monitoring.", 1),
    ("External HDD 2TB", 2190, "Portable hard drive for backup and storage expansion.", 1),
    ("USB-C to HDMI Adapter", 350, "Compact converter for connecting laptop to TV.", 1),
    ("Cleaning Kit 7-in-1", 250, "Multi-functional cleaning tool for screens and buds.", 1),
    ("Laptop Cooling Pad", 690, "Dual fan cooling stand for gaming laptops.", 1),
    ("Phone Desk Stand", 190, "Foldable aluminum stand for smartphones and tablets.", 1),
    ("Multi-plug Power Strip", 590, "6-outlet surge protector with 4 USB charging ports.", 1),
    ("Trackball Mouse", 1490, "Wireless trackball mouse for precise thumb control.", 1),
    ("Wired Earphones", 390, "Classic in-ear headphones with 3.5mm jack and mic.", 1),
    ("Monitor Light Bar", 1390, "Screen hanging light to reduce eye strain and glare.", 1),
    ("Thermal Paste", 290, "High-performance cooling grease for CPU and GPU.", 1),
    ("Joystick Controller", 990, "Classic style USB joystick for arcade gaming.", 1),
    ("Smart Home Camera", 1290, "1080p indoor security camera with night vision.", 1),
    ("Bluetooth Car Adapter", 390, "FM transmitter and hands-free car kit.", 1),
    ("Travel Adapter Universal", 450, "All-in-one international plug adapter for 150+ countries.", 1),
    ("VR Headset Starter", 1590, "Smartphone-based VR goggles for 3D movies and games.", 0),
    ("Anti-glare Screen Filter", 490, "Privacy screen protector for 15.6 inch laptops.", 1),
    ("Mechanical Num Pad", 790, "Separate 21-key numeric keypad with blue switches.", 1),
    ("USB-C Wall Charger 65W", 990, "GaN fast charger for laptops and smartphones.", 1),
    ("Car Phone Mount", 290, "Magnetic dashboard mount for easy phone access.", 1),
    ("Digital Voice Recorder", 1190, "Portable recorder with 8GB internal storage.", 1),
    ("Webcam Cover Slider", 90, "Ultra-thin privacy cover for laptop cameras (3-pack).", 1),
    ("Gaming Controller Wireless", 1890, "Bluetooth gamepad compatible with PC, Switch, and Mobile.", 1),
    ("Tablet Protective Case", 450, "Slim folio cover with auto wake/sleep feature.", 1),
    ("Mini PC Windows 11", 7990, "Compact desktop computer with 8GB RAM and 256GB SSD.", 1),
    ("Smart Water Bottle", 890, "Hydration tracker with LED glow reminders.", 0),
    ("Portable Label Maker", 1290, "Bluetooth label printer for home and office organization.", 1),
    ("Keyboard Wrist Rest", 350, "Memory foam support for typing comfort.", 1),
    ("USB Sound Card", 320, "External audio adapter with 3.5mm jack.", 1),
    ("Fitness Tracker Band", 990, "Slim activity tracker with step and sleep tracking.", 1),
    ("Desktop Fan USB", 290, "Small quiet fan for your workspace.", 1),
    ("Electric Screwdriver Kit", 1390, "Precision tool set for electronics repair.", 1),
    ("Smart Door Lock", 4590, "Keyless entry lock with fingerprint and app control.", 1),
    ("WiFi Range Extender", 890, "Plug-in signal booster for dead zones.", 1),
    ("Laptop Privacy Filter", 750, "Magnetic screen protector for MacBook Air.", 1),
    ("USB Fingerprint Reader", 650, "Biometric login scanner for Windows Hello.", 1),
    ("Gaming Glasses", 590, "Blue light blocking glasses for long gaming sessions.", 1),
    ("Portable Monitor 15.6", 4990, "USB-C travel monitor for dual-screen productivity.", 1),
    ("Smart Scales", 1190, "Body fat scale with app tracking and Bluetooth.", 1),
    ("Dimmable LED Strip", 450, "5m RGB light strip with remote and phone control.", 1),
    ("Earbud Case Silicone", 150, "Protective cover for wireless earbud cases.", 1),
    ("Wireless Desktop Set", 1290, "Full-size keyboard and mouse combo with one receiver.", 1),
    ("USB Floppy Drive", 490, "External 3.5-inch diskette drive for legacy media.", 0),
    ("Noise Isolating Earplugs", 390, "Soft silicone plugs for sleep and travel.", 1),
    ("Smart Pet Feeder", 3490, "Automatic food dispenser with camera and WiFi.", 1),
    ("Action Camera 4K", 2190, "Waterproof sports camera with accessory kit.", 1),
    ("Mini Humidifier USB", 350, "Portable cool mist humidifier for small rooms.", 1),
    ("Laptop Docking Station", 3290, "Dual 4K display dock with 10 ports.", 1),
    ("Phone Lens Kit", 490, "Clip-on macro and wide-angle lenses for mobile.", 1),
    ("Smart Notebook", 890, "Reusable digital notebook with cloud sync.", 1),
    ("Electric Air Pump", 1290, "Portable tire inflator for cars and bikes.", 1),
    ("USB-C Flash Drive 64GB", 390, "Dual-connector drive for phone and computer.", 1),
    ("Gaming Steering Wheel", 4590, "Force feedback racing wheel for driving sims.", 1),
    ("Wireless Doorbell", 550, "Battery-operated chime with 300m range.", 1),
    ("Handheld Game Console", 1890, "Retro gaming device with 500 built-in games.", 1),
    ("Solar Power Bank", 1190, "Rugged battery pack with solar charging panel.", 1),
    ("Smart Temperature Sensor", 450, "WiFi monitor for home humidity and temp.", 1),
    ("Mechanical Blue Switches", 490, "Pack of 90 replacement keyboard switches.", 1),
    ("Laptop Privacy Shutter", 120, "Physical slider for webcam privacy.", 1),
    ("USB 3.0 Hub 4-port", 390, "Slim data hub for laptop expansion.", 1),
    ("Gaming Microphone Arm", 1190, "Adjustable boom arm for studio mics.", 1),
    ("Smart Tag Tracker", 890, "Bluetooth finder for keys and wallets.", 1),
    ("Electric Foot Warmer", 990, "Heated mat for under-desk comfort.", 0),
    ("Desktop Organizer", 450, "Multi-compartment storage for office supplies.", 1),
    ("Wireless Guitar System", 2490, "Digital transmitter for electric instruments.", 1),
    ("Smart Mirror LED", 1890, "Wall mirror with touch controls and anti-fog.", 1),
]


@dataclass(frozen=True)
class Product:
    id: int
    name: str
    price_baht: int
    description: str
    in_stock: bool

    def to_summary(self) -> str:
        stock = "in stock" if self.in_stock else "out of stock"
        return f"{self.name} — {self.price_baht} baht ({stock}). {self.description}"


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    """Create the products table and seed it if empty. Safe to call repeatedly."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price_baht INTEGER NOT NULL,
                description TEXT NOT NULL,
                in_stock INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        if count == 0:
            conn.executemany(
                "INSERT INTO products (name, price_baht, description, in_stock) "
                "VALUES (?, ?, ?, ?)",
                _SEED_PRODUCTS,
            )
        conn.commit()


def _row_to_product(row: sqlite3.Row) -> Product:
    return Product(
        id=row["id"],
        name=row["name"],
        price_baht=row["price_baht"],
        description=row["description"],
        in_stock=bool(row["in_stock"]),
    )


def search_products(db_path: str | Path, query: str, limit: int = 5) -> list[Product]:
    """Case-insensitive substring search on name and description."""
    pattern = f"%{query.strip()}%"
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM products "
            "WHERE name LIKE ? COLLATE NOCASE OR description LIKE ? COLLATE NOCASE "
            "ORDER BY in_stock DESC, name ASC LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
    return [_row_to_product(r) for r in rows]


def list_all_products(db_path: str | Path) -> list[Product]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM products ORDER BY in_stock DESC, name ASC"
        ).fetchall()
    return [_row_to_product(r) for r in rows]

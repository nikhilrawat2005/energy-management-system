import json

PAGES = {
    'page1': {
        'file': 'config/grafana/provisioning/dashboards/page1_overview.json',
        'active': 'page1',
        'title': '1. MAEMS - Executive Microgrid Overview'
    },
    'page2': {
        'file': 'config/grafana/provisioning/dashboards/page2_weather_forecast.json',
        'active': 'page2',
        'title': '2. MAEMS - Weather & Forecasting Intelligence'
    },
    'page3': {
        'file': 'config/grafana/provisioning/dashboards/page3_battery_grid.json',
        'active': 'page3',
        'title': '3. MAEMS - Battery & Grid Diagnostics'
    },
    'page4': {
        'file': 'config/grafana/provisioning/dashboards/page4_agents_audit.json',
        'active': 'page4',
        'title': '4. MAEMS - Multi-Agent Decisions & Safety Audit'
    }
}

NAV_ITEMS = [
    ('page1', '⚡ 1. Microgrid Overview', '/d/maems_dashboard_v1/1-maems-executive-microgrid-overview'),
    ('page2', '🌤️ 2. Weather & Forecasts', '/d/maems_weather_v1/2-maems-weather-and-forecasting-intelligence'),
    ('page3', '🔋 3. Battery & Grid Diagnostics', '/d/maems_battery_grid_v1/3-maems-battery-and-grid-diagnostics'),
    ('page4', '🤖 4. Multi-Agent Decisions', '/d/maems_agents_v1/4-maems-multi-agent-decisions-and-safety-audit')
]

for key, meta in PAGES.items():
    with open(meta['file'], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build sleek HTML navigation bar buttons
    buttons_html = []
    for nav_key, nav_title, nav_url in NAV_ITEMS:
        is_active = (nav_key == meta['active'])
        if is_active:
            bg = '#1f6feb'
            border = '#388bfd'
            color = '#ffffff'
            weight = 'bold'
        else:
            bg = '#21262d'
            border = '#30363d'
            color = '#c9d1d9'
            weight = 'normal'
            
        btn = f'<a href="{nav_url}" target="_self" style="display: inline-block; padding: 7px 18px; margin-right: 12px; background-color: {bg}; border: 1px solid {border}; border-radius: 6px; color: {color}; font-weight: {weight}; text-decoration: none; font-size: 14px; font-family: Inter, Helvetica, Arial, sans-serif;">{nav_title}</a>'
        buttons_html.append(btn)

    nav_content = '<div style="display: flex; align-items: center; background: #0d1117; padding: 10px 14px; border-radius: 8px; border: 1px solid #30363d; width: 100%; box-sizing: border-box;">' + ''.join(buttons_html) + '</div>'

    nav_panel = {
        'id': 999,
        'type': 'text',
        'title': '',
        'gridPos': {'h': 2, 'w': 24, 'x': 0, 'y': 0},
        'options': {
            'mode': 'html',
            'content': nav_content
        },
        'transparent': True
    }

    # Remove any existing 999 panel and shift panels
    existing = [p for p in data.get('panels', []) if p.get('id') != 999]
    for p in existing:
        p['gridPos']['y'] = p['gridPos'].get('y', 0) + 2

    data['panels'] = [nav_panel] + existing

    with open(meta['file'], 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

print('Added embedded interactive HTML navigation bar to all 4 dashboards!')

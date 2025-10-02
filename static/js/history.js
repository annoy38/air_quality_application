document.getElementById("ping").onclick = async ()=>{
  try{
    const r = await fetch("http://localhost:8000/api/forecast");
    const j = await r.json();
    out.textContent = JSON.stringify(j,null,2);
  }catch(e){ out.textContent = e.message; }
};
